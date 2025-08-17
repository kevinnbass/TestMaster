"""
LLM Integration for Hierarchical Test Planning

Connects the Universal LLM Provider to the Hierarchical Test Planning system
for intelligent test generation with multi-provider fallback capabilities.
"""

import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..llm_providers.universal_llm_provider import (
    LLMProviderManager, 
    LLMMessage, 
    MessageRole,
    LLMResponse
)
from .htp_reasoning import PlanGenerator, PlanEvaluator, PlanningNode, EvaluationCriteria
from ...core.shared_state import cache_llm_response, get_cached_llm_response


@dataclass
class LLMPlanningConfig:
    """Configuration for LLM-powered planning."""
    primary_provider: str = "gemini"
    fallback_providers: List[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000
    enable_caching: bool = True
    planning_model: str = "gemini-pro"
    evaluation_model: str = "gemini-pro"


class LLMPoweredPlanGenerator(PlanGenerator):
    """Plan generator that uses LLMs to create test plans."""
    
    def __init__(self, config: LLMPlanningConfig = None):
        self.config = config or LLMPlanningConfig()
        self.llm_manager = LLMProviderManager()
        self._setup_providers()
        
        print("LLM-Powered Plan Generator initialized")
        print(f"   Primary provider: {self.config.primary_provider}")
        print(f"   Caching enabled: {self.config.enable_caching}")
    
    def _setup_providers(self):
        """Setup LLM providers with fallback chain."""
        # This would be configured based on available providers
        # For now, we'll set up a basic configuration
        fallback_order = self.config.fallback_providers or ["openai", "anthropic", "local"]
        self.llm_manager.set_fallback_order(fallback_order)
    
    def generate(self, parent_node: PlanningNode, context: Dict[str, Any]) -> List[PlanningNode]:
        """Generate child test plans using LLM reasoning."""
        
        # Extract planning context
        module_path = context.get('module_path', '')
        module_analysis = context.get('module_analysis', {})
        
        # Create planning prompt
        prompt = self._create_planning_prompt(parent_node, module_analysis, context)
        
        # Check cache first
        if self.config.enable_caching:
            cached_response = get_cached_llm_response(prompt, self.config.planning_model)
            if cached_response:
                print("Using cached LLM planning response")
                return self._parse_llm_response(cached_response, parent_node)
        
        # Generate with LLM
        try:
            llm_response = self._query_llm(prompt)
            
            # Cache the response
            if self.config.enable_caching and llm_response.content:
                cache_llm_response(prompt, llm_response.content, self.config.planning_model)
            
            # Parse response into planning nodes
            children = self._parse_llm_response(llm_response.content, parent_node)
            
            print(f"LLM generated {len(children)} test planning options")
            return children
            
        except Exception as e:
            print(f"LLM planning failed: {e}")
            # Fallback to template-based generation
            return self._fallback_generation(parent_node, context)
    
    def _create_planning_prompt(self, parent_node: PlanningNode, 
                               module_analysis: Dict[str, Any], 
                               context: Dict[str, Any]) -> str:
        """Create a prompt for LLM-based test planning."""
        
        module_path = context.get('module_path', 'unknown')
        parent_plan = parent_node.content
        
        prompt = f"""You are an expert test planning assistant. Generate detailed hierarchical test plans for a Python module.

MODULE TO TEST: {module_path}

MODULE ANALYSIS:
- Functions: {len(module_analysis.get('functions', []))}
- Classes: {len(module_analysis.get('classes', []))}
- Complexity Score: {module_analysis.get('complexity', 0)}
- Has Async: {module_analysis.get('has_async', False)}
- Has Exceptions: {module_analysis.get('has_exceptions', False)}

PARENT PLANNING CONTEXT:
{json.dumps(parent_plan, indent=2)}

REQUIREMENTS:
1. Generate 2-4 distinct test planning strategies
2. Each strategy should have a different focus (basic, comprehensive, security, performance)
3. Include estimated coverage, time, and quality metrics
4. Structure plans hierarchically with multiple levels

OUTPUT FORMAT (JSON):
{{
    "strategies": [
        {{
            "name": "strategy_name",
            "description": "brief description",
            "estimated_coverage": 0.85,
            "estimated_time": 45.0,
            "quality_score": 0.9,
            "test_types": ["unit", "integration", "security"],
            "levels": [
                {{
                    "level": 1,
                    "name": "level_name",
                    "description": "what this level tests",
                    "test_count": 5,
                    "coverage_target": 0.7,
                    "test_types": ["unit"]
                }}
            ]
        }}
    ]
}}

Focus on creating practical, executable test plans that balance coverage, efficiency, and quality."""
        
        return prompt
    
    def _query_llm(self, prompt: str) -> LLMResponse:
        """Query the LLM with the planning prompt."""
        
        messages = [
            LLMMessage(
                role=MessageRole.SYSTEM,
                content="You are an expert software test planning assistant. Generate detailed, practical test plans in the requested JSON format."
            ),
            LLMMessage(
                role=MessageRole.USER,
                content=prompt
            )
        ]
        
        # Use synchronous call for simplicity
        response = self.llm_manager.generate_sync(
            messages,
            preferred_provider=self.config.primary_provider,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response
    
    def _parse_llm_response(self, response_content: str, parent_node: PlanningNode) -> List[PlanningNode]:
        """Parse LLM response into planning nodes."""
        children = []
        
        try:
            # Extract JSON from response
            response_data = self._extract_json(response_content)
            strategies = response_data.get('strategies', [])
            
            for i, strategy in enumerate(strategies):
                # Create planning node for this strategy
                child_node = PlanningNode(
                    id=f"{parent_node.id}_llm_plan_{i}",
                    content={
                        'strategy': strategy.get('name', f'strategy_{i}'),
                        'description': strategy.get('description', ''),
                        'estimated_coverage': strategy.get('estimated_coverage', 0.7),
                        'estimated_time': strategy.get('estimated_time', 60.0),
                        'quality_score': strategy.get('quality_score', 0.8),
                        'test_types': strategy.get('test_types', []),
                        'levels': strategy.get('levels', []),
                        'generated_by': 'llm',
                        'model_used': self.config.planning_model
                    },
                    depth=parent_node.depth + 1
                )
                
                children.append(child_node)
        
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            # Return empty list - fallback will be handled by caller
            
        return children
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Try to find JSON block
        start_markers = ['```json', '```', '{']
        end_markers = ['```', '}']
        
        # Find JSON content
        for start in start_markers:
            if start in text:
                start_idx = text.find(start)
                if start == '{':
                    json_start = start_idx
                else:
                    json_start = start_idx + len(start)
                
                # Find matching end
                if start == '{':
                    # Count braces to find matching closing brace
                    brace_count = 0
                    for i, char in enumerate(text[json_start:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_text = text[json_start:json_start + i + 1]
                                return json.loads(json_text)
                else:
                    for end in end_markers:
                        end_idx = text.find(end, json_start)
                        if end_idx != -1:
                            json_text = text[json_start:end_idx].strip()
                            return json.loads(json_text)
        
        # If no JSON found, try parsing the entire text
        return json.loads(text)
    
    def _fallback_generation(self, parent_node: PlanningNode, context: Dict[str, Any]) -> List[PlanningNode]:
        """Fallback to template-based generation when LLM fails."""
        print("Falling back to template-based plan generation")
        
        from .test_plan_generator import TestPlanGenerator
        fallback_generator = TestPlanGenerator()
        return fallback_generator.generate(parent_node, context)


class LLMPoweredPlanEvaluator(PlanEvaluator):
    """Plan evaluator that uses LLMs to assess test plan quality."""
    
    def __init__(self, config: LLMPlanningConfig = None):
        self.config = config or LLMPlanningConfig()
        self.llm_manager = LLMProviderManager()
        
        print("LLM-Powered Plan Evaluator initialized")
    
    def evaluate(self, node: PlanningNode, criteria: List[EvaluationCriteria]) -> float:
        """Evaluate a test planning node using LLM assessment."""
        
        plan = node.content
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(plan, criteria)
        
        # Check cache first
        if self.config.enable_caching:
            cached_response = get_cached_llm_response(prompt, self.config.evaluation_model)
            if cached_response:
                return self._parse_evaluation_response(cached_response, node)
        
        try:
            # Get LLM evaluation
            llm_response = self._query_llm_for_evaluation(prompt)
            
            # Cache the response
            if self.config.enable_caching and llm_response.content:
                cache_llm_response(prompt, llm_response.content, self.config.evaluation_model)
            
            # Parse evaluation score
            score = self._parse_evaluation_response(llm_response.content, node)
            return score
            
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            # Fallback to rule-based evaluation
            return self._fallback_evaluation(node, criteria)
    
    def _create_evaluation_prompt(self, plan: Dict[str, Any], criteria: List[EvaluationCriteria]) -> str:
        """Create a prompt for LLM-based plan evaluation."""
        
        criteria_descriptions = []
        for criterion in criteria:
            criteria_descriptions.append(f"- {criterion.name} (weight: {criterion.weight}): {criterion.description}")
        
        prompt = f"""Evaluate this test plan across multiple criteria and provide detailed scoring.

TEST PLAN:
{json.dumps(plan, indent=2)}

EVALUATION CRITERIA:
{chr(10).join(criteria_descriptions)}

Please evaluate the test plan on each criterion (0.0 to 1.0 scale) and provide reasoning.

OUTPUT FORMAT (JSON):
{{
    "overall_score": 0.85,
    "criterion_scores": {{
        "coverage_potential": 0.9,
        "time_efficiency": 0.8,
        "quality_score": 0.85,
        "implementation_feasibility": 0.9,
        "strategic_value": 0.8
    }},
    "reasoning": "Brief explanation of the evaluation",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "recommendations": ["rec1", "rec2"]
}}

Consider factors like:
- Realistic coverage targets
- Appropriate time estimates
- Feasibility of implementation
- Strategic value for testing goals
- Balance between thoroughness and efficiency"""
        
        return prompt
    
    def _query_llm_for_evaluation(self, prompt: str) -> LLMResponse:
        """Query LLM for plan evaluation."""
        
        messages = [
            LLMMessage(
                role=MessageRole.SYSTEM,
                content="You are an expert software testing evaluator. Assess test plans objectively and provide detailed scoring with reasoning."
            ),
            LLMMessage(
                role=MessageRole.USER,
                content=prompt
            )
        ]
        
        response = self.llm_manager.generate_sync(
            messages,
            preferred_provider=self.config.primary_provider,
            temperature=0.1,  # Lower temperature for evaluation consistency
            max_tokens=1500
        )
        
        return response
    
    def _parse_evaluation_response(self, response_content: str, node: PlanningNode) -> float:
        """Parse LLM evaluation response."""
        
        try:
            # Extract JSON from response
            evaluation_data = self._extract_json(response_content)
            
            # Store individual criterion scores
            criterion_scores = evaluation_data.get('criterion_scores', {})
            for criterion, score in criterion_scores.items():
                node.update_score(criterion, score)
            
            # Store evaluation metadata
            node.metadata.update({
                'llm_evaluation': {
                    'reasoning': evaluation_data.get('reasoning', ''),
                    'strengths': evaluation_data.get('strengths', []),
                    'weaknesses': evaluation_data.get('weaknesses', []),
                    'recommendations': evaluation_data.get('recommendations', [])
                }
            })
            
            return evaluation_data.get('overall_score', 0.5)
            
        except Exception as e:
            print(f"Failed to parse LLM evaluation: {e}")
            return 0.5  # Default score
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Reuse the same logic as in the generator
        start_markers = ['```json', '```', '{']
        end_markers = ['```', '}']
        
        for start in start_markers:
            if start in text:
                start_idx = text.find(start)
                if start == '{':
                    json_start = start_idx
                else:
                    json_start = start_idx + len(start)
                
                if start == '{':
                    brace_count = 0
                    for i, char in enumerate(text[json_start:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_text = text[json_start:json_start + i + 1]
                                return json.loads(json_text)
                else:
                    for end in end_markers:
                        end_idx = text.find(end, json_start)
                        if end_idx != -1:
                            json_text = text[json_start:end_idx].strip()
                            return json.loads(json_text)
        
        return json.loads(text)
    
    def _fallback_evaluation(self, node: PlanningNode, criteria: List[EvaluationCriteria]) -> float:
        """Fallback to rule-based evaluation when LLM fails."""
        print("Falling back to rule-based plan evaluation")
        
        from .test_plan_generator import TestPlanEvaluator
        fallback_evaluator = TestPlanEvaluator()
        return fallback_evaluator.evaluate(node, criteria)


def create_llm_powered_planner(config: LLMPlanningConfig = None) -> 'HierarchicalTestPlanner':
    """Create a hierarchical test planner powered by LLMs."""
    from .htp_reasoning import HierarchicalTestPlanner, PlanningStrategy
    
    config = config or LLMPlanningConfig()
    
    generator = LLMPoweredPlanGenerator(config)
    evaluator = LLMPoweredPlanEvaluator(config)
    
    planner = HierarchicalTestPlanner(
        plan_generator=generator,
        plan_evaluator=evaluator,
        strategy=PlanningStrategy.BEST_FIRST,
        max_depth=4,
        max_iterations=30,  # Lower for LLM efficiency
        beam_width=3
    )
    
    # Add evaluation criteria
    planner.add_criterion(EvaluationCriteria(
        name="coverage_potential",
        weight=0.3,
        description="Potential for achieving high test coverage"
    ))
    
    planner.add_criterion(EvaluationCriteria(
        name="time_efficiency",
        weight=0.2,
        description="Efficiency of test generation time"
    ))
    
    planner.add_criterion(EvaluationCriteria(
        name="quality_score",
        weight=0.25,
        description="Expected quality of generated tests"
    ))
    
    planner.add_criterion(EvaluationCriteria(
        name="implementation_feasibility",
        weight=0.15,
        description="Feasibility of implementing the plan"
    ))
    
    planner.add_criterion(EvaluationCriteria(
        name="strategic_value",
        weight=0.1,
        description="Strategic value for testing objectives"
    ))
    
    print("LLM-Powered Hierarchical Test Planner created")
    return planner


def test_llm_integration():
    """Test the LLM integration with hierarchical planning."""
    print("\n" + "="*60)
    print("Testing LLM Integration with Hierarchical Planning")
    print("="*60)
    
    # Create test context
    test_context = {
        'module_path': 'test_module.py',
        'module_analysis': {
            'functions': ['func1', 'func2', 'func3'],
            'classes': ['Class1'],
            'complexity': 12,
            'has_async': True,
            'has_exceptions': True,
            'source_code': 'def func1(): pass'
        }
    }
    
    # Create LLM-powered planner
    config = LLMPlanningConfig(
        primary_provider="gemini",
        enable_caching=True,
        temperature=0.3
    )
    
    try:
        planner = create_llm_powered_planner(config)
        
        # Create initial plan
        initial_plan = {
            'objective': 'generate_comprehensive_tests',
            'module_path': 'test_module.py',
            'requirements': ['high_coverage', 'security_tests'],
            'constraints': ['time_limit_120s']
        }
        
        # Execute planning (this would call LLM in real scenario)
        print("\nExecuting hierarchical planning...")
        planning_tree = planner.plan(initial_plan, test_context)
        
        # Get results
        best_plan = planning_tree.get_best_plan()
        stats = planning_tree.get_statistics()
        
        print(f"\nPlanning Results:")
        print(f"  Total nodes generated: {stats['total_nodes']}")
        print(f"  Max depth reached: {stats['max_depth']}")
        print(f"  Leaf nodes: {stats['leaf_nodes']}")
        print(f"  Best plan nodes: {len(best_plan)}")
        
        if best_plan:
            final_plan = best_plan[-1].content
            print(f"\nBest Plan Strategy: {final_plan.get('strategy', 'unknown')}")
            print(f"  Expected Coverage: {final_plan.get('estimated_coverage', 0):.1%}")
            print(f"  Estimated Time: {final_plan.get('estimated_time', 0):.1f}s")
            print(f"  Quality Score: {best_plan[-1].aggregate_score:.2f}")
        
        print("\n✅ LLM Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ LLM Integration test failed: {e}")
        return False


if __name__ == "__main__":
    test_llm_integration()