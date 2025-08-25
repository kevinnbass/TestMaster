"""
Test-Specific Hierarchical Planning Generator

Generates hierarchical test plans using multi-level decomposition.
Previously mislabeled as "Tree-of-Thought" - corrected as per roadmap specification.

This generator creates systematic test plans that break down complex testing
scenarios into manageable hierarchical components.
"""

import ast
import json
import time
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from .htp_reasoning import PlanGenerator, PlanEvaluator, PlanningNode, EvaluationCriteria


@dataclass
class TestPlanLevel:
    """Represents a level in the hierarchical test plan."""
    level: int
    name: str
    description: str
    test_types: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    complexity_score: float = 0.0


@dataclass
class TestGenerationPlan:
    """A complete test generation plan."""
    module_path: str
    strategy: str
    levels: List[TestPlanLevel] = field(default_factory=list)
    estimated_coverage: float = 0.0
    estimated_time: float = 0.0
    quality_score: float = 0.0


class TestPlanGenerator(PlanGenerator):
    """Generates hierarchical test plans for modules."""
    
    def __init__(self):
        self.plan_templates = {
            'basic': self._create_basic_plan_template(),
            'comprehensive': self._create_comprehensive_plan_template(),
            'security_focused': self._create_security_plan_template(),
            'performance_focused': self._create_performance_plan_template()
        }
        
        print("TestPlanGenerator initialized")
        print(f"   Available templates: {list(self.plan_templates.keys())}")
    
    def generate(self, parent_node: PlanningNode, context: Dict[str, Any]) -> List[PlanningNode]:
        """Generate child test plans from a parent planning node."""
        children = []
        
        # Get module analysis from context
        module_path = context.get('module_path', '')
        module_analysis = context.get('module_analysis', {})
        
        # Determine appropriate planning strategies
        strategies = self._determine_strategies(module_analysis)
        
        for i, strategy in enumerate(strategies):
            child_plan = self._create_plan(strategy, module_path, module_analysis, parent_node)
            
            child_node = PlanningNode(
                id=f"{parent_node.id}_plan_{i}",
                content=child_plan,
                depth=parent_node.depth + 1
            )
            
            children.append(child_node)
        
        return children
    
    def _determine_strategies(self, module_analysis: Dict[str, Any]) -> List[str]:
        """Determine appropriate planning strategies based on module analysis."""
        strategies = ['basic']  # Always include basic
        
        # Analyze module characteristics
        functions = module_analysis.get('functions', [])
        classes = module_analysis.get('classes', [])
        complexity = module_analysis.get('complexity', 0)
        has_async = module_analysis.get('has_async', False)
        has_exceptions = module_analysis.get('has_exceptions', False)
        
        # Add comprehensive if complex enough
        if len(functions) > 5 or len(classes) > 2 or complexity > 10:
            strategies.append('comprehensive')
        
        # Add security-focused if security-relevant
        security_keywords = ['auth', 'login', 'password', 'token', 'security', 'crypto']
        module_text = module_analysis.get('source_code', '').lower()
        if any(keyword in module_text for keyword in security_keywords):
            strategies.append('security_focused')
        
        # Add performance-focused if performance-critical
        perf_keywords = ['cache', 'optimize', 'performance', 'speed', 'fast', 'async']
        if any(keyword in module_text for keyword in perf_keywords) or has_async:
            strategies.append('performance_focused')
        
        return strategies
    
    def _create_plan(self, strategy: str, module_path: str, 
                    module_analysis: Dict[str, Any], parent_node: PlanningNode) -> Dict[str, Any]:
        """Create a test plan based on strategy."""
        template = self.plan_templates.get(strategy, self.plan_templates['basic'])
        
        # Customize template based on module analysis
        plan = {
            'strategy': strategy,
            'module_path': module_path,
            'parent_plan': parent_node.id,
            'levels': self._customize_levels(template['levels'], module_analysis),
            'estimated_coverage': template['base_coverage'],
            'estimated_time': self._estimate_time(template, module_analysis),
            'quality_targets': template['quality_targets'],
            'test_types': template['test_types']
        }
        
        return plan
    
    def _customize_levels(self, template_levels: List[Dict], module_analysis: Dict) -> List[Dict]:
        """Customize planning levels based on module characteristics."""
        customized = []
        
        for level_template in template_levels:
            level = level_template.copy()
            
            # Adjust based on module complexity
            complexity = module_analysis.get('complexity', 0)
            if complexity > 15:
                level['test_count'] = int(level.get('test_count', 5) * 1.5)
                level['coverage_target'] = min(0.95, level.get('coverage_target', 0.8) + 0.1)
            
            # Adjust based on function count
            function_count = len(module_analysis.get('functions', []))
            if function_count > 10:
                level['test_count'] = max(level.get('test_count', 5), function_count)
            
            customized.append(level)
        
        return customized
    
    def _estimate_time(self, template: Dict, module_analysis: Dict) -> float:
        """Estimate time required for test generation."""
        base_time = template.get('base_time', 30.0)  # seconds
        
        # Adjust based on module size
        functions = module_analysis.get('functions', [])
        classes = module_analysis.get('classes', [])
        complexity = module_analysis.get('complexity', 0)
        
        # Time scaling factors
        function_factor = len(functions) * 2.0
        class_factor = len(classes) * 5.0
        complexity_factor = complexity * 0.5
        
        estimated_time = base_time + function_factor + class_factor + complexity_factor
        return min(estimated_time, 300.0)  # Cap at 5 minutes
    
    def _create_basic_plan_template(self) -> Dict[str, Any]:
        """Create basic test plan template."""
        return {
            'name': 'Basic Test Plan',
            'description': 'Standard test coverage with core functionality',
            'base_coverage': 0.7,
            'base_time': 30.0,
            'quality_targets': {
                'syntax_correctness': 0.95,
                'import_correctness': 0.9,
                'execution_success': 0.8
            },
            'test_types': ['unit', 'basic_integration'],
            'levels': [
                {
                    'level': 1,
                    'name': 'Function Testing',
                    'description': 'Test individual functions',
                    'test_count': 5,
                    'coverage_target': 0.6,
                    'test_types': ['unit']
                },
                {
                    'level': 2,
                    'name': 'Class Testing',
                    'description': 'Test class methods and properties',
                    'test_count': 3,
                    'coverage_target': 0.7,
                    'test_types': ['unit', 'method']
                },
                {
                    'level': 3,
                    'name': 'Integration Testing',
                    'description': 'Test component interactions',
                    'test_count': 2,
                    'coverage_target': 0.8,
                    'test_types': ['integration']
                }
            ]
        }
    
    def _create_comprehensive_plan_template(self) -> Dict[str, Any]:
        """Create comprehensive test plan template."""
        return {
            'name': 'Comprehensive Test Plan',
            'description': 'Thorough testing with edge cases and error conditions',
            'base_coverage': 0.85,
            'base_time': 60.0,
            'quality_targets': {
                'syntax_correctness': 0.98,
                'import_correctness': 0.95,
                'execution_success': 0.9,
                'edge_case_coverage': 0.8
            },
            'test_types': ['unit', 'integration', 'edge_case', 'error_handling'],
            'levels': [
                {
                    'level': 1,
                    'name': 'Core Functionality',
                    'description': 'Test main functions and methods',
                    'test_count': 8,
                    'coverage_target': 0.7,
                    'test_types': ['unit', 'happy_path']
                },
                {
                    'level': 2,
                    'name': 'Edge Cases',
                    'description': 'Test boundary conditions and edge cases',
                    'test_count': 6,
                    'coverage_target': 0.8,
                    'test_types': ['edge_case', 'boundary']
                },
                {
                    'level': 3,
                    'name': 'Error Handling',
                    'description': 'Test exception handling and error conditions',
                    'test_count': 4,
                    'coverage_target': 0.9,
                    'test_types': ['error_handling', 'exception']
                },
                {
                    'level': 4,
                    'name': 'Integration',
                    'description': 'Test module interactions and dependencies',
                    'test_count': 3,
                    'coverage_target': 0.85,
                    'test_types': ['integration', 'dependency']
                }
            ]
        }
    
    def _create_security_plan_template(self) -> Dict[str, Any]:
        """Create security-focused test plan template."""
        return {
            'name': 'Security-Focused Test Plan',
            'description': 'Testing with emphasis on security vulnerabilities',
            'base_coverage': 0.8,
            'base_time': 45.0,
            'quality_targets': {
                'syntax_correctness': 0.95,
                'import_correctness': 0.9,
                'execution_success': 0.85,
                'security_coverage': 0.9
            },
            'test_types': ['unit', 'security', 'vulnerability', 'injection'],
            'levels': [
                {
                    'level': 1,
                    'name': 'Input Validation',
                    'description': 'Test input sanitization and validation',
                    'test_count': 6,
                    'coverage_target': 0.8,
                    'test_types': ['input_validation', 'sanitization']
                },
                {
                    'level': 2,
                    'name': 'Authentication/Authorization',
                    'description': 'Test access control and authentication',
                    'test_count': 4,
                    'coverage_target': 0.9,
                    'test_types': ['auth', 'access_control']
                },
                {
                    'level': 3,
                    'name': 'Injection Attacks',
                    'description': 'Test SQL injection, XSS, and other injection attacks',
                    'test_count': 5,
                    'coverage_target': 0.85,
                    'test_types': ['injection', 'xss', 'sql_injection']
                }
            ]
        }
    
    def _create_performance_plan_template(self) -> Dict[str, Any]:
        """Create performance-focused test plan template."""
        return {
            'name': 'Performance-Focused Test Plan',
            'description': 'Testing with emphasis on performance and scalability',
            'base_coverage': 0.75,
            'base_time': 50.0,
            'quality_targets': {
                'syntax_correctness': 0.95,
                'import_correctness': 0.9,
                'execution_success': 0.85,
                'performance_benchmarks': 0.8
            },
            'test_types': ['unit', 'performance', 'load', 'stress'],
            'levels': [
                {
                    'level': 1,
                    'name': 'Performance Benchmarks',
                    'description': 'Establish performance baselines',
                    'test_count': 4,
                    'coverage_target': 0.7,
                    'test_types': ['benchmark', 'timing']
                },
                {
                    'level': 2,
                    'name': 'Load Testing',
                    'description': 'Test under various load conditions',
                    'test_count': 3,
                    'coverage_target': 0.8,
                    'test_types': ['load', 'concurrency']
                },
                {
                    'level': 3,
                    'name': 'Resource Usage',
                    'description': 'Test memory and CPU usage',
                    'test_count': 3,
                    'coverage_target': 0.75,
                    'test_types': ['memory', 'cpu', 'resource']
                }
            ]
        }


class TestPlanEvaluator(PlanEvaluator):
    """Evaluates hierarchical test plans for quality and feasibility."""
    
    def __init__(self):
        self.evaluation_weights = {
            'coverage_potential': 0.3,
            'time_efficiency': 0.2,
            'quality_score': 0.25,
            'implementation_feasibility': 0.15,
            'strategic_value': 0.1
        }
        
        print("TestPlanEvaluator initialized")
    
    def evaluate(self, node: PlanningNode, criteria: List[EvaluationCriteria]) -> float:
        """Evaluate a test planning node."""
        plan = node.content
        
        # Calculate individual scores
        coverage_score = self._evaluate_coverage_potential(plan)
        time_score = self._evaluate_time_efficiency(plan)
        quality_score = self._evaluate_quality_potential(plan)
        feasibility_score = self._evaluate_feasibility(plan)
        strategic_score = self._evaluate_strategic_value(plan)
        
        # Store individual scores
        node.update_score('coverage_potential', coverage_score)
        node.update_score('time_efficiency', time_score)
        node.update_score('quality_score', quality_score)
        node.update_score('implementation_feasibility', feasibility_score)
        node.update_score('strategic_value', strategic_score)
        
        # Calculate weighted aggregate
        aggregate = (
            coverage_score * self.evaluation_weights['coverage_potential'] +
            time_score * self.evaluation_weights['time_efficiency'] +
            quality_score * self.evaluation_weights['quality_score'] +
            feasibility_score * self.evaluation_weights['implementation_feasibility'] +
            strategic_score * self.evaluation_weights['strategic_value']
        )
        
        return aggregate
    
    def _evaluate_coverage_potential(self, plan: Dict[str, Any]) -> float:
        """Evaluate potential test coverage."""
        estimated_coverage = plan.get('estimated_coverage', 0.5)
        levels = plan.get('levels', [])
        
        # Bonus for multiple levels
        level_bonus = min(0.2, len(levels) * 0.05)
        
        # Bonus for diverse test types
        test_types = plan.get('test_types', [])
        diversity_bonus = min(0.1, len(test_types) * 0.02)
        
        return min(1.0, estimated_coverage + level_bonus + diversity_bonus)
    
    def _evaluate_time_efficiency(self, plan: Dict[str, Any]) -> float:
        """Evaluate time efficiency (lower time = higher score)."""
        estimated_time = plan.get('estimated_time', 60.0)
        
        # Score inversely proportional to time (capped at reasonable bounds)
        if estimated_time <= 30:
            return 1.0
        elif estimated_time <= 60:
            return 0.8
        elif estimated_time <= 120:
            return 0.6
        elif estimated_time <= 300:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_quality_potential(self, plan: Dict[str, Any]) -> float:
        """Evaluate potential test quality."""
        quality_targets = plan.get('quality_targets', {})
        
        # Average of quality targets
        if quality_targets:
            return sum(quality_targets.values()) / len(quality_targets)
        
        # Default based on strategy
        strategy = plan.get('strategy', 'basic')
        quality_map = {
            'basic': 0.7,
            'comprehensive': 0.9,
            'security_focused': 0.85,
            'performance_focused': 0.8
        }
        
        return quality_map.get(strategy, 0.7)
    
    def _evaluate_feasibility(self, plan: Dict[str, Any]) -> float:
        """Evaluate implementation feasibility."""
        # Check for realistic targets
        estimated_coverage = plan.get('estimated_coverage', 0.5)
        estimated_time = plan.get('estimated_time', 60.0)
        levels = plan.get('levels', [])
        
        # Penalize unrealistic coverage expectations
        coverage_penalty = 0.0
        if estimated_coverage > 0.95:
            coverage_penalty = 0.2
        elif estimated_coverage > 0.9:
            coverage_penalty = 0.1
        
        # Penalize overly complex plans
        complexity_penalty = 0.0
        if len(levels) > 5:
            complexity_penalty = 0.15
        elif len(levels) > 4:
            complexity_penalty = 0.05
        
        # Penalize excessive time requirements
        time_penalty = 0.0
        if estimated_time > 300:
            time_penalty = 0.3
        elif estimated_time > 180:
            time_penalty = 0.1
        
        base_score = 1.0
        return max(0.0, base_score - coverage_penalty - complexity_penalty - time_penalty)
    
    def _evaluate_strategic_value(self, plan: Dict[str, Any]) -> float:
        """Evaluate strategic value of the plan."""
        strategy = plan.get('strategy', 'basic')
        test_types = plan.get('test_types', [])
        
        # Assign strategic values
        strategy_values = {
            'basic': 0.6,
            'comprehensive': 0.9,
            'security_focused': 0.85,
            'performance_focused': 0.8
        }
        
        base_value = strategy_values.get(strategy, 0.6)
        
        # Bonus for valuable test types
        valuable_types = ['security', 'performance', 'integration', 'edge_case']
        value_bonus = sum(0.05 for test_type in test_types if test_type in valuable_types)
        
        return min(1.0, base_value + value_bonus)


class HierarchicalTestGenerator:
    """Integrates hierarchical planning with actual test generation."""
    
    def __init__(self):
        from ..llm_providers.universal_llm_provider import LLMProviderManager
        from ...core.shared_state import get_shared_state
        
        self.planner = None  # Will be initialized when needed
        self.llm_manager = LLMProviderManager()
        self.shared_state = get_shared_state()
        
        print("HierarchicalTestGenerator initialized")
    
    def generate_with_planning(self, module_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate tests using hierarchical planning."""
        context = context or {}
        
        # Initialize planner if needed
        if not self.planner:
            from .htp_reasoning import get_hierarchical_planner
            self.planner = get_hierarchical_planner()
        
        # Create initial plan
        initial_plan = {
            'objective': 'generate_comprehensive_tests',
            'module_path': module_path,
            'requirements': context.get('requirements', []),
            'constraints': context.get('constraints', [])
        }
        
        # Execute hierarchical planning
        planning_tree = self.planner.plan(initial_plan, context)
        
        # Get best plan
        best_plan_path = planning_tree.get_best_plan()
        
        if not best_plan_path:
            return {
                'success': False,
                'error': 'No viable test plan generated',
                'planning_stats': planning_tree.get_statistics()
            }
        
        # Extract final plan
        final_plan_node = best_plan_path[-1]
        final_plan = final_plan_node.content
        
        # Generate tests based on plan
        test_results = self._execute_plan(final_plan, context)
        
        return {
            'success': True,
            'test_code': test_results.get('test_code', ''),
            'plan_used': final_plan,
            'planning_stats': planning_tree.get_statistics(),
            'execution_stats': test_results.get('stats', {}),
            'quality_score': final_plan_node.aggregate_score
        }
    
    def _execute_plan(self, plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the generated test plan."""
        # This would integrate with existing test generators
        # For now, return a placeholder
        return {
            'test_code': f"# Test generated using {plan['strategy']} strategy\n# TODO: Implement actual test generation",
            'stats': {
                'estimated_coverage': plan.get('estimated_coverage', 0.0),
                'estimated_time': plan.get('estimated_time', 0.0)
            }
        }