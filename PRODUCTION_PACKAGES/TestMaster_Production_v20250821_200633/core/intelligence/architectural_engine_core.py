"""
Architectural Engine Core - Streamlined Architectural Decision Engine
====================================================================

Streamlined core orchestration engine implementing comprehensive architectural
decision making, multi-criteria analysis, and intelligent recommendation generation
with enterprise-grade decision support and implementation planning.

This module provides the core architectural decision framework including:
- Unified architectural decision orchestration
- Multi-criteria decision analysis coordination
- Microservice evolution strategy generation
- Implementation planning and risk assessment
- Evidence-based recommendation engine

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: architectural_engine_core.py (320 lines)
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import uuid

from .architectural_types import (
    DecisionType, ArchitecturalOption, DecisionContext, DecisionAnalysis,
    OptionEvaluation, MicroserviceMetrics, ServiceBoundary, MigrationStrategy,
    PatternRecommendation, DecisionPriority
)
from .decision_analyzer import DecisionAnalyzer
from .microservice_analyzer import DomainBoundaryAnalyzer, MicroserviceEvolutionAnalyzer

logger = logging.getLogger(__name__)


class TradeOffAnalysisEngine:
    """
    Advanced trade-off analysis engine for evaluating architectural decisions
    with comprehensive Pareto optimization and multi-objective analysis.
    """
    
    def __init__(self):
        self.trade_off_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("TradeOffAnalysisEngine initialized")
    
    async def analyze_trade_offs(
        self, evaluations: List[OptionEvaluation], context: DecisionContext
    ) -> Dict[str, Any]:
        """
        Perform comprehensive trade-off analysis of architectural options.
        
        Args:
            evaluations: List of option evaluations
            context: Decision context
            
        Returns:
            Comprehensive trade-off analysis with Pareto frontier
        """
        logger.info(f"Analyzing trade-offs for {len(evaluations)} options")
        
        analysis = {
            "pareto_frontier": await self._calculate_pareto_frontier(evaluations),
            "trade_off_matrix": self._generate_trade_off_matrix(evaluations),
            "sensitivity_analysis": await self._perform_sensitivity_analysis(evaluations, context),
            "risk_return_analysis": self._analyze_risk_return_profile(evaluations),
            "recommendation_rationale": self._generate_recommendation_rationale(evaluations, context),
            "decision_confidence": self._calculate_decision_confidence(evaluations)
        }
        
        logger.info("Trade-off analysis completed")
        return analysis
    
    async def _calculate_pareto_frontier(self, evaluations: List[OptionEvaluation]) -> List[Dict[str, Any]]:
        """Calculate Pareto frontier for multi-objective optimization"""
        
        pareto_options = []
        
        for evaluation in evaluations:
            is_pareto_optimal = True
            
            for other_evaluation in evaluations:
                if other_evaluation == evaluation:
                    continue
                
                # Check if other option dominates this one
                if self._dominates(other_evaluation, evaluation):
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_options.append({
                    "option_id": evaluation.option.id,
                    "option_name": evaluation.option.name,
                    "overall_score": evaluation.overall_score,
                    "weighted_score": evaluation.weighted_score,
                    "rank": evaluation.rank,
                    "key_strengths": self._identify_key_strengths(evaluation),
                    "key_weaknesses": self._identify_key_weaknesses(evaluation)
                })
        
        return sorted(pareto_options, key=lambda x: x["weighted_score"], reverse=True)
    
    def _dominates(self, eval1: OptionEvaluation, eval2: OptionEvaluation) -> bool:
        """Check if eval1 Pareto-dominates eval2"""
        
        # Extract scores by criteria
        scores1 = {cs.criteria: cs.score for cs in eval1.criteria_scores}
        scores2 = {cs.criteria: cs.score for cs in eval2.criteria_scores}
        
        # Check if eval1 is better or equal in all criteria and strictly better in at least one
        better_in_all = True
        strictly_better_in_one = False
        
        for criteria in scores1.keys():
            if criteria in scores2:
                if scores1[criteria] < scores2[criteria]:
                    better_in_all = False
                    break
                elif scores1[criteria] > scores2[criteria]:
                    strictly_better_in_one = True
        
        return better_in_all and strictly_better_in_one
    
    def _generate_trade_off_matrix(self, evaluations: List[OptionEvaluation]) -> Dict[str, Dict[str, float]]:
        """Generate trade-off matrix showing option performance across criteria"""
        
        matrix = {}
        
        for evaluation in evaluations:
            option_scores = {}
            for cs in evaluation.criteria_scores:
                option_scores[cs.criteria.value] = cs.score
            matrix[evaluation.option.name] = option_scores
        
        return matrix
    
    async def _perform_sensitivity_analysis(
        self, evaluations: List[OptionEvaluation], context: DecisionContext
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on criteria weights"""
        
        sensitivity = {
            "weight_variations": {},
            "ranking_stability": {},
            "critical_factors": []
        }
        
        # Test weight variations
        base_rankings = [eval.rank for eval in sorted(evaluations, key=lambda x: x.rank)]
        
        for eval in evaluations:
            for cs in eval.criteria_scores:
                criteria_name = cs.criteria.value
                
                # Test +/- 20% weight variation
                variations = []
                for weight_change in [-0.2, -0.1, 0.1, 0.2]:
                    modified_evaluations = self._apply_weight_change(
                        evaluations, cs.criteria, weight_change
                    )
                    new_rankings = [eval.rank for eval in sorted(modified_evaluations, key=lambda x: x.weighted_score, reverse=True)]
                    rank_changes = sum(abs(b - n) for b, n in zip(base_rankings, new_rankings))
                    variations.append(rank_changes)
                
                sensitivity["weight_variations"][criteria_name] = {
                    "average_rank_change": sum(variations) / len(variations),
                    "max_rank_change": max(variations)
                }
        
        # Identify critical factors (high sensitivity)
        critical_threshold = 2.0
        for criteria, data in sensitivity["weight_variations"].items():
            if data["average_rank_change"] > critical_threshold:
                sensitivity["critical_factors"].append(criteria)
        
        return sensitivity
    
    def _apply_weight_change(
        self, evaluations: List[OptionEvaluation], criteria, weight_change: float
    ) -> List[OptionEvaluation]:
        """Apply weight change and recalculate scores"""
        
        modified_evaluations = []
        
        for evaluation in evaluations:
            new_criteria_scores = []
            total_weight = 0
            
            for cs in evaluation.criteria_scores:
                if cs.criteria == criteria:
                    new_weight = max(0.01, cs.weight + weight_change)
                else:
                    new_weight = cs.weight
                
                new_criteria_scores.append(type(cs)(
                    criteria=cs.criteria,
                    score=cs.score,
                    weight=new_weight,
                    rationale=cs.rationale,
                    confidence=cs.confidence
                ))
                total_weight += new_weight
            
            # Normalize weights
            for cs in new_criteria_scores:
                cs.weight /= total_weight
            
            # Recalculate weighted score
            new_weighted_score = sum(cs.score * cs.weight for cs in new_criteria_scores)
            
            modified_eval = type(evaluation)(
                option=evaluation.option,
                criteria_scores=new_criteria_scores,
                overall_score=evaluation.overall_score,
                weighted_score=new_weighted_score,
                rank=evaluation.rank,
                recommendation_strength=evaluation.recommendation_strength,
                implementation_feasibility=evaluation.implementation_feasibility,
                risk_assessment=evaluation.risk_assessment,
                trade_offs=evaluation.trade_offs
            )
            
            modified_evaluations.append(modified_eval)
        
        return modified_evaluations
    
    def _analyze_risk_return_profile(self, evaluations: List[OptionEvaluation]) -> Dict[str, Any]:
        """Analyze risk-return profile of options"""
        
        risk_return = {
            "options": [],
            "risk_categories": ["low", "medium", "high"],
            "return_categories": ["low", "medium", "high"]
        }
        
        for evaluation in evaluations:
            overall_risk = sum(evaluation.risk_assessment.values()) / len(evaluation.risk_assessment) if evaluation.risk_assessment else 0.5
            overall_return = evaluation.weighted_score
            
            risk_return["options"].append({
                "name": evaluation.option.name,
                "risk": overall_risk,
                "return": overall_return,
                "risk_category": "high" if overall_risk > 0.7 else "medium" if overall_risk > 0.4 else "low",
                "return_category": "high" if overall_return > 0.8 else "medium" if overall_return > 0.6 else "low",
                "risk_adjusted_return": overall_return / (1 + overall_risk)
            })
        
        return risk_return
    
    def _generate_recommendation_rationale(
        self, evaluations: List[OptionEvaluation], context: DecisionContext
    ) -> str:
        """Generate recommendation rationale based on analysis"""
        
        if not evaluations:
            return "No options available for analysis"
        
        best_option = min(evaluations, key=lambda x: x.rank)
        
        rationale = f"Recommended option: {best_option.option.name}\n\n"
        rationale += f"Overall score: {best_option.weighted_score:.3f}\n"
        rationale += f"Implementation feasibility: {best_option.implementation_feasibility:.2f}\n\n"
        
        rationale += "Key strengths:\n"
        strengths = self._identify_key_strengths(best_option)
        for strength in strengths[:3]:
            rationale += f"- {strength}\n"
        
        if best_option.trade_offs:
            rationale += "\nKey trade-offs:\n"
            for trade_off in best_option.trade_offs[:3]:
                rationale += f"- {trade_off}\n"
        
        return rationale
    
    def _identify_key_strengths(self, evaluation: OptionEvaluation) -> List[str]:
        """Identify key strengths of an option"""
        
        strengths = []
        
        # High-scoring criteria
        high_scores = [cs for cs in evaluation.criteria_scores if cs.score > 0.8]
        for cs in high_scores:
            strengths.append(f"Excellent {cs.criteria.value}: {cs.rationale}")
        
        # Implementation feasibility
        if evaluation.implementation_feasibility > 0.8:
            strengths.append("High implementation feasibility")
        
        return strengths
    
    def _identify_key_weaknesses(self, evaluation: OptionEvaluation) -> List[str]:
        """Identify key weaknesses of an option"""
        
        weaknesses = []
        
        # Low-scoring criteria
        low_scores = [cs for cs in evaluation.criteria_scores if cs.score < 0.5]
        for cs in low_scores:
            weaknesses.append(f"Weak {cs.criteria.value}: {cs.rationale}")
        
        # High risks
        high_risks = {k: v for k, v in evaluation.risk_assessment.items() if v > 0.7}
        for risk_type, risk_level in high_risks.items():
            weaknesses.append(f"High {risk_type} risk: {risk_level:.2f}")
        
        return weaknesses
    
    def _calculate_decision_confidence(self, evaluations: List[OptionEvaluation]) -> float:
        """Calculate confidence level in the decision"""
        
        if len(evaluations) < 2:
            return 0.5
        
        # Sort by weighted score
        sorted_evals = sorted(evaluations, key=lambda x: x.weighted_score, reverse=True)
        
        # Calculate gap between top two options
        gap = sorted_evals[0].weighted_score - sorted_evals[1].weighted_score
        
        # Higher gap = higher confidence
        confidence = min(0.5 + gap * 2, 0.95)
        
        return confidence


class ArchitecturalDecisionEngine:
    """
    Streamlined architectural decision engine implementing comprehensive
    decision making, microservice evolution, and intelligent recommendations.
    
    Features:
    - Multi-criteria architectural decision analysis
    - Microservice boundary optimization and evolution
    - Trade-off analysis with Pareto optimization  
    - Implementation planning and risk assessment
    - Evidence-based recommendation generation
    """
    
    def __init__(self):
        # Core analysis engines
        self.decision_analyzer = DecisionAnalyzer()
        self.boundary_analyzer = DomainBoundaryAnalyzer()
        self.evolution_analyzer = MicroserviceEvolutionAnalyzer()
        self.trade_off_engine = TradeOffAnalysisEngine()
        
        # Decision state
        self.decision_history: List[DecisionAnalysis] = []
        self.active_contexts: Dict[str, DecisionContext] = {}
        
        logger.info("ArchitecturalDecisionEngine initialized")
    
    async def make_architectural_decision(
        self, decision_type: DecisionType, options: List[ArchitecturalOption],
        context: DecisionContext
    ) -> DecisionAnalysis:
        """
        Make comprehensive architectural decision with full analysis.
        
        Args:
            decision_type: Type of architectural decision
            options: Available architectural options
            context: Decision context with requirements and constraints
            
        Returns:
            Complete decision analysis with recommendation
        """
        logger.info(f"Making architectural decision: {decision_type.value}")
        
        decision_id = f"ARCH-{uuid.uuid4().hex[:8].upper()}"
        start_time = time.time()
        
        try:
            # Phase 1: Analyze all options
            logger.info("Phase 1: Analyzing architectural options")
            evaluations = await self.decision_analyzer.analyze_options(options, context)
            
            # Phase 2: Perform trade-off analysis
            logger.info("Phase 2: Performing trade-off analysis")
            trade_off_analysis = await self.trade_off_engine.analyze_trade_offs(evaluations, context)
            
            # Phase 3: Generate implementation plan
            logger.info("Phase 3: Generating implementation plan")
            recommended_option = evaluations[0] if evaluations else None
            implementation_plan = []
            if recommended_option:
                implementation_plan = await self._generate_implementation_plan(
                    recommended_option.option, context
                )
            
            # Phase 4: Define success metrics
            success_metrics = self._define_success_metrics(decision_type, recommended_option)
            
            # Calculate review date
            review_date = datetime.now() + timedelta(days=90)
            
            # Create decision analysis
            analysis = DecisionAnalysis(
                id=decision_id,
                decision_type=decision_type,
                context=context,
                options=options,
                evaluations=evaluations,
                recommended_option=recommended_option.option if recommended_option else None,
                decision_rationale=trade_off_analysis.get("recommendation_rationale", "No clear recommendation"),
                confidence_level=trade_off_analysis.get("decision_confidence", 0.5),
                implementation_plan=implementation_plan,
                success_metrics=success_metrics,
                review_date=review_date
            )
            
            # Store decision
            self.decision_history.append(analysis)
            self.active_contexts[decision_id] = context
            
            execution_time = time.time() - start_time
            logger.info(f"Architectural decision completed in {execution_time:.2f}s")
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error making architectural decision: {e}")
            # Return minimal analysis on error
            return DecisionAnalysis(
                id=decision_id,
                decision_type=decision_type,
                context=context,
                options=options,
                evaluations=[],
                recommended_option=None,
                decision_rationale=f"Decision analysis failed: {str(e)}",
                confidence_level=0.0
            )
    
    async def analyze_microservice_evolution(
        self, current_architecture: Dict[str, Any], target_requirements: Dict[str, Any],
        context: DecisionContext
    ) -> Dict[str, Any]:
        """
        Analyze microservice architecture evolution strategy.
        
        Args:
            current_architecture: Current architecture description
            target_requirements: Target architecture requirements
            context: Decision context
            
        Returns:
            Complete microservice evolution analysis
        """
        logger.info("Analyzing microservice evolution strategy")
        
        try:
            # Phase 1: Analyze service boundaries
            logger.info("Phase 1: Analyzing service boundaries")
            service_boundaries = await self.boundary_analyzer.analyze_service_boundaries(
                current_architecture, context
            )
            
            # Phase 2: Create current metrics
            current_metrics = self._extract_microservice_metrics(current_architecture)
            
            # Phase 3: Analyze evolution strategy
            logger.info("Phase 2: Analyzing evolution strategy")
            evolution_analysis = await self.evolution_analyzer.analyze_evolution_strategy(
                current_metrics, target_requirements, context
            )
            
            # Combine results
            complete_analysis = {
                "service_boundaries": service_boundaries,
                "current_metrics": current_metrics,
                "evolution_analysis": evolution_analysis,
                "recommendations": self._generate_microservice_recommendations(
                    service_boundaries, evolution_analysis
                ),
                "implementation_roadmap": self._create_implementation_roadmap(
                    evolution_analysis
                )
            }
            
            logger.info("Microservice evolution analysis completed")
            return complete_analysis
        
        except Exception as e:
            logger.error(f"Error analyzing microservice evolution: {e}")
            return {"error": str(e), "analysis_failed": True}
    
    async def _generate_implementation_plan(
        self, option: ArchitecturalOption, context: DecisionContext
    ) -> List[str]:
        """Generate detailed implementation plan"""
        
        plan = [
            "1. Architecture Design Phase",
            "   - Create detailed system design",
            "   - Define component interfaces",
            "   - Establish technical standards"
        ]
        
        # Add pattern-specific steps
        for pattern in option.patterns:
            pattern_steps = self._get_pattern_steps(pattern.value)
            plan.extend([f"   - {step}" for step in pattern_steps])
        
        plan.extend([
            "2. Infrastructure Setup",
            "   - Prepare development environment",
            "   - Set up CI/CD pipelines",
            "   - Configure monitoring systems",
            "",
            "3. Implementation Phase", 
            "   - Implement core components",
            "   - Integration testing",
            "   - Performance optimization",
            "",
            "4. Deployment and Validation",
            "   - Staged deployment",
            "   - Production validation",
            "   - Performance monitoring"
        ])
        
        return plan
    
    def _get_pattern_steps(self, pattern_name: str) -> List[str]:
        """Get implementation steps for architectural pattern"""
        
        pattern_steps = {
            "microservices": ["Define service boundaries", "Implement service communication", "Set up service discovery"],
            "serverless": ["Design function boundaries", "Configure event triggers", "Set up API Gateway"],
            "event_driven": ["Design event schemas", "Implement event handlers", "Set up event streaming"]
        }
        
        return pattern_steps.get(pattern_name, ["Pattern-specific implementation"])
    
    def _define_success_metrics(
        self, decision_type: DecisionType, evaluation: Optional[OptionEvaluation]
    ) -> List[str]:
        """Define success metrics for architectural decision"""
        
        metrics = []
        
        if decision_type == DecisionType.MICROSERVICE_BOUNDARIES:
            metrics = [
                "Service coupling reduced by 30%",
                "Deployment frequency increased by 2x",
                "Cross-service latency < 100ms"
            ]
        elif decision_type == DecisionType.SCALING_STRATEGY:
            metrics = [
                "Auto-scaling response time < 60s",
                "Cost per transaction reduced by 20%",
                "99.9% availability maintained"
            ]
        elif decision_type == DecisionType.TECHNOLOGY_SELECTION:
            metrics = [
                "Developer productivity maintained",
                "Performance requirements met",
                "Technical debt does not increase"
            ]
        else:
            metrics = [
                "Implementation completed on time",
                "Performance requirements satisfied", 
                "Team adoption successful"
            ]
        
        return metrics
    
    def _extract_microservice_metrics(self, architecture: Dict[str, Any]) -> MicroserviceMetrics:
        """Extract microservice metrics from architecture description"""
        
        services = architecture.get("services", [])
        
        return MicroserviceMetrics(
            service_count=len(services),
            average_service_size=50.0,  # Would calculate from actual data
            coupling_score=0.5,  # Would analyze service dependencies
            cohesion_score=0.7,  # Would analyze internal cohesion
            api_complexity=0.6,  # Would analyze API complexity
            data_consistency_level=0.8,  # Would analyze data patterns
            deployment_complexity=0.5,  # Would analyze deployment patterns
            communication_overhead=0.4  # Would analyze communication patterns
        )
    
    def _generate_microservice_recommendations(
        self, boundaries: List[ServiceBoundary], evolution_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate microservice-specific recommendations"""
        
        recommendations = []
        
        recommendations.append(f"Identified {len(boundaries)} optimal service boundaries")
        
        # Add evolution-specific recommendations
        evolution_patterns = evolution_analysis.get("evolution_patterns", [])
        if evolution_patterns:
            top_pattern = evolution_patterns[0]
            recommendations.append(f"Recommended evolution pattern: {top_pattern.pattern.value}")
        
        # Add migration strategy recommendation
        migration_strategy = evolution_analysis.get("migration_strategy")
        if migration_strategy:
            recommendations.append(f"Recommended migration: {migration_strategy.migration_type}")
        
        return recommendations
    
    def _create_implementation_roadmap(self, evolution_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create implementation roadmap from evolution analysis"""
        
        roadmap = {
            "phases": [],
            "timeline": "12-18 months",
            "key_milestones": []
        }
        
        migration_strategy = evolution_analysis.get("migration_strategy")
        if migration_strategy:
            roadmap["phases"] = migration_strategy.phases
            roadmap["timeline"] = f"{migration_strategy.timeline.days // 7} weeks"
        
        return roadmap


# Export architectural engine components
__all__ = ['ArchitecturalDecisionEngine', 'TradeOffAnalysisEngine']