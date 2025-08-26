"""
Cross-Framework Integration Module
=================================
Advanced integration system for unified documentation across all 7 frameworks.

This module provides seamless integration and cross-pollination of documentation
patterns from Agency-Swarm, CrewAI, AgentScope, AutoGen, LLama-Agents, PhiData, and Swarms.

Author: Agent D - Documentation Intelligence
"""

from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from datetime import datetime
import asyncio

class FrameworkType(Enum):
    """Supported framework types for documentation integration."""
    AGENCY_SWARM = "agency_swarm"
    CREWAI = "crewai"
    AGENTSCOPE = "agentscope"
    AUTOGEN = "autogen"
    LLAMA_AGENTS = "llama_agents"
    PHIDATA = "phidata"
    SWARMS = "swarms"

class IntegrationType(Enum):
    """Types of cross-framework integrations."""
    PATTERN_FUSION = "pattern_fusion"
    HYBRID_APPROACH = "hybrid_approach"
    BEST_OF_BREED = "best_of_breed"
    COMPLEMENTARY = "complementary"
    COMPARATIVE = "comparative"

@dataclass
class FrameworkPattern:
    """Represents a documentation pattern from a specific framework."""
    framework: FrameworkType
    pattern_name: str
    pattern_type: str
    description: str
    code_examples: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    compatibility_matrix: Dict[str, bool] = field(default_factory=dict)
    quality_score: float = 0.0

@dataclass
class IntegrationOpportunity:
    """Represents a cross-framework integration opportunity."""
    primary_framework: FrameworkType
    secondary_frameworks: List[FrameworkType]
    integration_type: IntegrationType
    compatibility_score: float
    benefits: List[str]
    challenges: List[str]
    implementation_complexity: str  # "low", "medium", "high"
    recommended_approach: str

@dataclass
class IntegratedDocumentationPlan:
    """Complete plan for integrated documentation generation."""
    target_frameworks: List[FrameworkType]
    integration_strategy: str
    pattern_combinations: List[Dict[str, Any]]
    expected_benefits: List[str]
    implementation_timeline: Dict[str, str]
    resource_requirements: Dict[str, Any]

class FrameworkCompatibilityAnalyzer:
    """Analyzes compatibility between different framework patterns."""
    
    def __init__(self):
        self.compatibility_cache: Dict[Tuple[FrameworkType, FrameworkType], float] = {}
        self.pattern_synergies = self._initialize_pattern_synergies()
    
    def _initialize_pattern_synergies(self) -> Dict[str, Dict[str, float]]:
        """Initialize known synergies between framework patterns."""
        return {
            "agent_patterns": {
                "agency_swarm_crewai": 0.9,      # Both use agent-based architectures
                "autogen_swarms": 0.8,           # Both support multi-agent conversations
                "llama_agents_crewai": 0.7,      # Both have workflow management
                "agentscope_autogen": 0.6        # Both have development tools
            },
            "workflow_patterns": {
                "crewai_llama_agents": 0.9,      # Both excel in workflows
                "agency_swarm_swarms": 0.7,      # Both have orchestration
                "autogen_phidata": 0.6           # Both have execution patterns
            },
            "documentation_patterns": {
                "phidata_agentscope": 0.8,       # Both have excellent cookbook patterns
                "autogen_crewai": 0.7,           # Both have comprehensive guides
                "agency_swarm_llama_agents": 0.6  # Both have deployment docs
            }
        }
    
    async def analyze_compatibility(self, framework_a: FrameworkType, 
                                  framework_b: FrameworkType) -> float:
        """Analyze compatibility between two frameworks."""
        cache_key = (framework_a, framework_b)
        
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]
        
        # Multi-dimensional compatibility analysis
        architectural_compatibility = self._assess_architectural_compatibility(framework_a, framework_b)
        pattern_compatibility = self._assess_pattern_compatibility(framework_a, framework_b)
        documentation_compatibility = self._assess_documentation_compatibility(framework_a, framework_b)
        
        overall_compatibility = (
            architectural_compatibility * 0.4 +
            pattern_compatibility * 0.4 +
            documentation_compatibility * 0.2
        )
        
        self.compatibility_cache[cache_key] = overall_compatibility
        return overall_compatibility
    
    def _assess_architectural_compatibility(self, framework_a: FrameworkType, 
                                          framework_b: FrameworkType) -> float:
        """Assess architectural compatibility between frameworks."""
        architecture_patterns = {
            FrameworkType.AGENCY_SWARM: ["agent_based", "tool_integration", "swarm_intelligence"],
            FrameworkType.CREWAI: ["agent_based", "workflow_orchestration", "task_management"],
            FrameworkType.AGENTSCOPE: ["studio_interface", "development_tools", "project_management"],
            FrameworkType.AUTOGEN: ["conversation_patterns", "multi_agent", "streaming"],
            FrameworkType.LLAMA_AGENTS: ["service_oriented", "workflow_management", "deployment"],
            FrameworkType.PHIDATA: ["cookbook_patterns", "data_visualization", "example_driven"],
            FrameworkType.SWARMS: ["swarm_intelligence", "coordination", "parallel_processing"]
        }
        
        patterns_a = set(architecture_patterns.get(framework_a, []))
        patterns_b = set(architecture_patterns.get(framework_b, []))
        
        if not patterns_a or not patterns_b:
            return 0.5  # Default compatibility
        
        intersection = patterns_a.intersection(patterns_b)
        union = patterns_a.union(patterns_b)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _assess_pattern_compatibility(self, framework_a: FrameworkType, 
                                    framework_b: FrameworkType) -> float:
        """Assess pattern-level compatibility."""
        # Check for known synergies
        for pattern_type, synergies in self.pattern_synergies.items():
            key_ab = f"{framework_a.value}_{framework_b.value}"
            key_ba = f"{framework_b.value}_{framework_a.value}"
            
            if key_ab in synergies:
                return synergies[key_ab]
            elif key_ba in synergies:
                return synergies[key_ba]
        
        # Default pattern compatibility assessment
        return self._calculate_default_pattern_compatibility(framework_a, framework_b)
    
    def _assess_documentation_compatibility(self, framework_a: FrameworkType, 
                                          framework_b: FrameworkType) -> float:
        """Assess documentation style compatibility."""
        doc_styles = {
            FrameworkType.AGENCY_SWARM: ["detailed_examples", "tool_focused", "practical"],
            FrameworkType.CREWAI: ["workflow_diagrams", "multilingual", "enterprise"],
            FrameworkType.AGENTSCOPE: ["interactive", "tutorial_based", "beginner_friendly"],
            FrameworkType.AUTOGEN: ["conversation_examples", "streaming_demos", "technical"],
            FrameworkType.LLAMA_AGENTS: ["service_docs", "deployment_guides", "production"],
            FrameworkType.PHIDATA: ["cookbook_style", "visual_examples", "recipe_based"],
            FrameworkType.SWARMS: ["coordination_docs", "intelligence_patterns", "scalability"]
        }
        
        styles_a = set(doc_styles.get(framework_a, []))
        styles_b = set(doc_styles.get(framework_b, []))
        
        if not styles_a or not styles_b:
            return 0.6  # Default doc compatibility
        
        intersection = styles_a.intersection(styles_b)
        union = styles_a.union(styles_b)
        
        return len(intersection) / len(union) if union else 0.3
    
    def _calculate_default_pattern_compatibility(self, framework_a: FrameworkType, 
                                               framework_b: FrameworkType) -> float:
        """Calculate default compatibility when no specific synergy is known."""
        # Based on framework categorization
        agent_focused = {FrameworkType.AGENCY_SWARM, FrameworkType.CREWAI, 
                        FrameworkType.AUTOGEN, FrameworkType.SWARMS}
        workflow_focused = {FrameworkType.CREWAI, FrameworkType.LLAMA_AGENTS}
        development_focused = {FrameworkType.AGENTSCOPE, FrameworkType.PHIDATA}
        
        if framework_a in agent_focused and framework_b in agent_focused:
            return 0.7
        elif framework_a in workflow_focused and framework_b in workflow_focused:
            return 0.8
        elif framework_a in development_focused and framework_b in development_focused:
            return 0.6
        else:
            return 0.5

class CrossFrameworkIntegrationEngine:
    """Main engine for cross-framework documentation integration."""
    
    def __init__(self):
        self.compatibility_analyzer = FrameworkCompatibilityAnalyzer()
        self.framework_patterns: Dict[FrameworkType, List[FrameworkPattern]] = {}
        self.integration_cache: Dict[str, Any] = {}
    
    def register_framework_patterns(self, framework: FrameworkType, 
                                  patterns: List[FrameworkPattern]) -> None:
        """Register patterns for a specific framework."""
        self.framework_patterns[framework] = patterns
    
    async def find_integration_opportunities(self, 
                                           target_frameworks: List[FrameworkType]) -> List[IntegrationOpportunity]:
        """Find optimal integration opportunities across frameworks."""
        opportunities = []
        
        for primary in target_frameworks:
            for secondary_list in self._get_framework_combinations(target_frameworks, primary):
                compatibility_score = await self._calculate_group_compatibility(primary, secondary_list)
                
                if compatibility_score >= 0.6:  # Minimum viable compatibility
                    opportunity = await self._create_integration_opportunity(
                        primary, secondary_list, compatibility_score
                    )
                    opportunities.append(opportunity)
        
        # Sort by compatibility score and potential benefits
        return sorted(opportunities, key=lambda x: x.compatibility_score, reverse=True)
    
    async def create_integrated_documentation_plan(self, 
                                                 opportunities: List[IntegrationOpportunity],
                                                 priority_criteria: Dict[str, float]) -> IntegratedDocumentationPlan:
        """Create a comprehensive plan for integrated documentation."""
        # Select top opportunities based on criteria
        weighted_opportunities = self._weight_opportunities_by_criteria(opportunities, priority_criteria)
        selected_opportunities = weighted_opportunities[:5]  # Top 5 integrations
        
        # Extract target frameworks
        target_frameworks = set()
        for opp in selected_opportunities:
            target_frameworks.add(opp.primary_framework)
            target_frameworks.update(opp.secondary_frameworks)
        
        # Generate integration strategy
        strategy = self._generate_integration_strategy(selected_opportunities)
        
        # Create pattern combinations
        pattern_combinations = self._create_pattern_combinations(selected_opportunities)
        
        # Calculate benefits and requirements
        expected_benefits = self._calculate_expected_benefits(selected_opportunities)
        implementation_timeline = self._create_implementation_timeline(selected_opportunities)
        resource_requirements = self._estimate_resource_requirements(selected_opportunities)
        
        return IntegratedDocumentationPlan(
            target_frameworks=list(target_frameworks),
            integration_strategy=strategy,
            pattern_combinations=pattern_combinations,
            expected_benefits=expected_benefits,
            implementation_timeline=implementation_timeline,
            resource_requirements=resource_requirements
        )
    
    def _get_framework_combinations(self, frameworks: List[FrameworkType], 
                                  primary: FrameworkType) -> List[List[FrameworkType]]:
        """Generate all possible combinations with a primary framework."""
        others = [f for f in frameworks if f != primary]
        combinations = []
        
        # Single secondary framework
        for secondary in others:
            combinations.append([secondary])
        
        # Multiple secondary frameworks (up to 3)
        if len(others) >= 2:
            for i, f1 in enumerate(others):
                for f2 in others[i+1:]:
                    combinations.append([f1, f2])
        
        if len(others) >= 3:
            for i, f1 in enumerate(others):
                for j, f2 in enumerate(others[i+1:], i+1):
                    for f3 in others[j+1:]:
                        combinations.append([f1, f2, f3])
        
        return combinations
    
    async def _calculate_group_compatibility(self, primary: FrameworkType, 
                                           secondary_list: List[FrameworkType]) -> float:
        """Calculate compatibility score for a group of frameworks."""
        if not secondary_list:
            return 0.0
        
        # Calculate pairwise compatibilities
        compatibility_scores = []
        
        for secondary in secondary_list:
            score = await self.compatibility_analyzer.analyze_compatibility(primary, secondary)
            compatibility_scores.append(score)
        
        # Calculate inter-secondary compatibilities
        if len(secondary_list) > 1:
            for i, f1 in enumerate(secondary_list):
                for f2 in secondary_list[i+1:]:
                    score = await self.compatibility_analyzer.analyze_compatibility(f1, f2)
                    compatibility_scores.append(score)
        
        # Return weighted average (primary relationships weighted higher)
        primary_weight = 0.6
        secondary_weight = 0.4
        
        primary_scores = compatibility_scores[:len(secondary_list)]
        inter_secondary_scores = compatibility_scores[len(secondary_list):]
        
        primary_avg = sum(primary_scores) / len(primary_scores) if primary_scores else 0
        secondary_avg = sum(inter_secondary_scores) / len(inter_secondary_scores) if inter_secondary_scores else 0
        
        return primary_avg * primary_weight + secondary_avg * secondary_weight
    
    async def _create_integration_opportunity(self, primary: FrameworkType, 
                                            secondary_list: List[FrameworkType], 
                                            compatibility_score: float) -> IntegrationOpportunity:
        """Create a detailed integration opportunity."""
        # Determine integration type based on frameworks and compatibility
        integration_type = self._determine_integration_type(primary, secondary_list, compatibility_score)
        
        # Generate benefits and challenges
        benefits = self._generate_integration_benefits(primary, secondary_list, integration_type)
        challenges = self._generate_integration_challenges(primary, secondary_list, integration_type)
        
        # Assess implementation complexity
        complexity = self._assess_implementation_complexity(primary, secondary_list, integration_type)
        
        # Generate recommended approach
        approach = self._generate_recommended_approach(primary, secondary_list, integration_type, complexity)
        
        return IntegrationOpportunity(
            primary_framework=primary,
            secondary_frameworks=secondary_list,
            integration_type=integration_type,
            compatibility_score=compatibility_score,
            benefits=benefits,
            challenges=challenges,
            implementation_complexity=complexity,
            recommended_approach=approach
        )
    
    def _determine_integration_type(self, primary: FrameworkType, 
                                  secondary_list: List[FrameworkType], 
                                  compatibility_score: float) -> IntegrationType:
        """Determine the most appropriate integration type."""
        if compatibility_score >= 0.8:
            return IntegrationType.PATTERN_FUSION
        elif compatibility_score >= 0.7:
            return IntegrationType.HYBRID_APPROACH
        elif compatibility_score >= 0.65:
            return IntegrationType.BEST_OF_BREED
        elif compatibility_score >= 0.6:
            return IntegrationType.COMPLEMENTARY
        else:
            return IntegrationType.COMPARATIVE
    
    def _generate_integration_benefits(self, primary: FrameworkType, 
                                     secondary_list: List[FrameworkType], 
                                     integration_type: IntegrationType) -> List[str]:
        """Generate expected benefits from integration."""
        base_benefits = [
            "Enhanced documentation coverage",
            "Cross-framework pattern reuse",
            "Improved developer experience",
            "Reduced learning curve"
        ]
        
        # Add specific benefits based on integration type
        if integration_type == IntegrationType.PATTERN_FUSION:
            base_benefits.extend([
                "Seamless pattern integration",
                "Unified development workflow",
                "Optimal resource utilization"
            ])
        elif integration_type == IntegrationType.HYBRID_APPROACH:
            base_benefits.extend([
                "Best practices from multiple frameworks",
                "Flexible implementation options",
                "Adaptive documentation strategies"
            ])
        
        # Add framework-specific benefits
        framework_benefits = {
            FrameworkType.AGENCY_SWARM: ["Enhanced tool integration", "Swarm intelligence patterns"],
            FrameworkType.CREWAI: ["Advanced workflow orchestration", "Enterprise-grade features"],
            FrameworkType.PHIDATA: ["Rich visualization examples", "Cookbook methodologies"],
            FrameworkType.AUTOGEN: ["Advanced conversation patterns", "Real-time interactions"]
        }
        
        for framework in [primary] + secondary_list:
            base_benefits.extend(framework_benefits.get(framework, []))
        
        return list(set(base_benefits))  # Remove duplicates
    
    def _generate_integration_challenges(self, primary: FrameworkType, 
                                       secondary_list: List[FrameworkType], 
                                       integration_type: IntegrationType) -> List[str]:
        """Generate potential challenges for integration."""
        base_challenges = [
            "Complexity of managing multiple paradigms",
            "Potential performance overhead",
            "Increased maintenance burden"
        ]
        
        if integration_type == IntegrationType.PATTERN_FUSION:
            base_challenges.append("Deep integration complexity")
        elif integration_type == IntegrationType.COMPARATIVE:
            base_challenges.append("Limited actual integration benefits")
        
        return base_challenges
    
    def _assess_implementation_complexity(self, primary: FrameworkType, 
                                        secondary_list: List[FrameworkType], 
                                        integration_type: IntegrationType) -> str:
        """Assess implementation complexity level."""
        complexity_factors = len(secondary_list)
        
        if integration_type in [IntegrationType.PATTERN_FUSION, IntegrationType.HYBRID_APPROACH]:
            complexity_factors += 2
        elif integration_type == IntegrationType.COMPARATIVE:
            complexity_factors -= 1
        
        if complexity_factors <= 1:
            return "low"
        elif complexity_factors <= 3:
            return "medium"
        else:
            return "high"
    
    def _generate_recommended_approach(self, primary: FrameworkType, 
                                     secondary_list: List[FrameworkType], 
                                     integration_type: IntegrationType,
                                     complexity: str) -> str:
        """Generate recommended implementation approach."""
        if complexity == "low":
            return f"Direct integration of {primary.value} patterns with {secondary_list[0].value} enhancements"
        elif complexity == "medium":
            return f"Phased integration starting with {primary.value} as base, gradually incorporating elements from {', '.join(f.value for f in secondary_list)}"
        else:
            return f"Modular integration approach with {primary.value} as orchestrator and selective integration of specific patterns from secondary frameworks"
    
    def _weight_opportunities_by_criteria(self, opportunities: List[IntegrationOpportunity], 
                                        criteria: Dict[str, float]) -> List[IntegrationOpportunity]:
        """Weight opportunities by priority criteria."""
        weighted_opportunities = []
        
        for opp in opportunities:
            score = opp.compatibility_score * criteria.get("compatibility", 0.3)
            score += len(opp.benefits) * criteria.get("benefits", 0.3) / 10
            score += (3 - len(opp.challenges)) * criteria.get("low_risk", 0.2) / 3
            
            complexity_weights = {"low": 1.0, "medium": 0.7, "high": 0.4}
            score += complexity_weights.get(opp.implementation_complexity, 0.5) * criteria.get("feasibility", 0.2)
            
            # Create weighted tuple for sorting
            weighted_opportunities.append((score, opp))
        
        # Sort by weighted score and return opportunities
        return [opp for _, opp in sorted(weighted_opportunities, key=lambda x: x[0], reverse=True)]
    
    def _generate_integration_strategy(self, opportunities: List[IntegrationOpportunity]) -> str:
        """Generate overall integration strategy."""
        if not opportunities:
            return "No viable integration opportunities identified"
        
        primary_frameworks = set(opp.primary_framework for opp in opportunities)
        
        if len(primary_frameworks) == 1:
            primary = list(primary_frameworks)[0]
            return f"Hub strategy with {primary.value} as primary framework, integrating complementary patterns from other frameworks"
        else:
            return "Multi-hub strategy with specialized integration points for different use cases"
    
    def _create_pattern_combinations(self, opportunities: List[IntegrationOpportunity]) -> List[Dict[str, Any]]:
        """Create specific pattern combinations from opportunities."""
        combinations = []
        
        for i, opp in enumerate(opportunities):
            combination = {
                "id": f"combination_{i+1}",
                "primary_framework": opp.primary_framework.value,
                "secondary_frameworks": [f.value for f in opp.secondary_frameworks],
                "integration_type": opp.integration_type.value,
                "complexity": opp.implementation_complexity,
                "estimated_effort": self._estimate_effort(opp),
                "expected_outcome": self._describe_expected_outcome(opp)
            }
            combinations.append(combination)
        
        return combinations
    
    def _calculate_expected_benefits(self, opportunities: List[IntegrationOpportunity]) -> List[str]:
        """Calculate overall expected benefits from all integrations."""
        all_benefits = set()
        
        for opp in opportunities:
            all_benefits.update(opp.benefits)
        
        return list(all_benefits)
    
    def _create_implementation_timeline(self, opportunities: List[IntegrationOpportunity]) -> Dict[str, str]:
        """Create implementation timeline based on opportunities."""
        timeline = {}
        current_week = 1
        
        for i, opp in enumerate(opportunities):
            effort_weeks = self._estimate_effort_weeks(opp)
            timeline[f"integration_{i+1}"] = f"Week {current_week}-{current_week + effort_weeks - 1}"
            current_week += effort_weeks
        
        timeline["total_duration"] = f"{current_week - 1} weeks"
        return timeline
    
    def _estimate_resource_requirements(self, opportunities: List[IntegrationOpportunity]) -> Dict[str, Any]:
        """Estimate resource requirements for implementations."""
        total_complexity_score = sum(
            {"low": 1, "medium": 2, "high": 3}[opp.implementation_complexity] 
            for opp in opportunities
        )
        
        return {
            "estimated_dev_hours": total_complexity_score * 40,
            "required_expertise": ["Framework integration", "Documentation architecture", "Pattern analysis"],
            "testing_requirements": ["Cross-framework compatibility tests", "Integration tests", "Performance benchmarks"],
            "documentation_updates": f"{len(opportunities)} integration guides",
            "maintenance_overhead": f"{total_complexity_score * 10}% increase"
        }
    
    def _estimate_effort(self, opportunity: IntegrationOpportunity) -> str:
        """Estimate effort for implementing an integration."""
        complexity_efforts = {
            "low": "1-2 developer days",
            "medium": "1-2 developer weeks",
            "high": "3-4 developer weeks"
        }
        
        return complexity_efforts.get(opportunity.implementation_complexity, "Unknown effort")
    
    def _describe_expected_outcome(self, opportunity: IntegrationOpportunity) -> str:
        """Describe the expected outcome of an integration."""
        primary = opportunity.primary_framework.value
        secondary = ", ".join(f.value for f in opportunity.secondary_frameworks)
        
        return f"Enhanced {primary} documentation with integrated patterns from {secondary}, achieving {opportunity.integration_type.value} approach"
    
    def _estimate_effort_weeks(self, opportunity: IntegrationOpportunity) -> int:
        """Estimate effort in weeks."""
        complexity_weeks = {"low": 1, "medium": 2, "high": 3}
        base_weeks = complexity_weeks.get(opportunity.implementation_complexity, 2)
        
        # Add complexity for multiple secondary frameworks
        additional_weeks = max(0, len(opportunity.secondary_frameworks) - 1)
        
        return base_weeks + additional_weeks

# Global integration registry
_integration_registry = CrossFrameworkIntegrationEngine()

def get_integration_engine() -> CrossFrameworkIntegrationEngine:
    """Get the global integration engine instance."""
    return _integration_registry

async def analyze_framework_synergies(frameworks: List[str]) -> Dict[str, Any]:
    """High-level function to analyze synergies between frameworks."""
    framework_types = [FrameworkType(f) for f in frameworks if f in [ft.value for ft in FrameworkType]]
    
    if len(framework_types) < 2:
        return {"error": "At least 2 valid frameworks required for synergy analysis"}
    
    engine = get_integration_engine()
    opportunities = await engine.find_integration_opportunities(framework_types)
    
    return {
        "analyzed_frameworks": [f.value for f in framework_types],
        "integration_opportunities": len(opportunities),
        "top_opportunity": {
            "primary": opportunities[0].primary_framework.value if opportunities else None,
            "secondary": [f.value for f in opportunities[0].secondary_frameworks] if opportunities else None,
            "compatibility_score": opportunities[0].compatibility_score if opportunities else None
        } if opportunities else None,
        "recommended_integrations": [
            {
                "primary": opp.primary_framework.value,
                "secondary": [f.value for f in opp.secondary_frameworks],
                "type": opp.integration_type.value,
                "score": opp.compatibility_score
            }
            for opp in opportunities[:3]  # Top 3 recommendations
        ]
    }