"""
Architectural Decision Engine - Consolidated Modular Version
===========================================================

Main architectural decision engine that coordinates all modularized decision-making capabilities.
This is the consolidated version using the modularized components for better maintainability.

Original file: 2,388 lines → Now modularized into focused modules with this coordinating engine.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
import hashlib
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from .architectural_decision_modules import (
    DecisionType, DecisionCriteria, ArchitecturalPattern, DecisionPriority,
    ArchitecturalOption, DecisionContext, DecisionAnalysis, PatternEvolution, PerformanceMetrics,
    DecisionScorer, DesignPatternEvolutionEngine, PerformanceArchitectureOptimizer, MicroserviceEvolutionAnalyzer
)

logger = logging.getLogger(__name__)


class ArchitecturalDecisionEngine:
    """
    Main architectural decision engine that coordinates all decision-making capabilities
    """
    
    def __init__(self):
        self.decision_scorer = DecisionScorer()
        self.pattern_evolution_engine = DesignPatternEvolutionEngine()
        self.performance_optimizer = PerformanceArchitectureOptimizer()
        self.microservice_analyzer = MicroserviceEvolutionAnalyzer()
        self.decision_history: List[DecisionAnalysis] = []
        self.decision_cache: Dict[str, DecisionAnalysis] = {}
        
        logger.info("ArchitecturalDecisionEngine initialized with comprehensive analysis capabilities")
    
    async def make_architectural_decision(self, context: DecisionContext,
                                        options: List[ArchitecturalOption],
                                        current_metrics: Optional[PerformanceMetrics] = None,
                                        target_metrics: Optional[PerformanceMetrics] = None) -> DecisionAnalysis:
        """
        Make comprehensive architectural decision based on context and options
        """
        logger.info(f"Making architectural decision for: {context.description}")
        
        # Check cache first
        cache_key = self._generate_cache_key(context, options)
        if cache_key in self.decision_cache:
            logger.info("Returning cached decision analysis")
            return self.decision_cache[cache_key]
        
        # Score all options
        scored_options = []
        for option in options:
            score = self.decision_scorer.score_option(option, context)
            scored_options.append((score, option))
        
        # Sort by score (highest first)
        scored_options.sort(key=lambda x: x[0], reverse=True)
        
        recommended_option = scored_options[0][1]
        alternative_options = [option for _, option in scored_options[1:3]]  # Top 2 alternatives
        
        # Generate comprehensive analysis
        analysis = await self._generate_comprehensive_analysis(
            context, recommended_option, alternative_options, current_metrics, target_metrics
        )
        
        # Cache and store decision
        self.decision_cache[cache_key] = analysis
        self.decision_history.append(analysis)
        
        logger.info(f"Decision made: {recommended_option.name} (confidence: {analysis.confidence_score:.2f})")
        return analysis
    
    async def analyze_pattern_evolution(self, current_patterns: List[ArchitecturalPattern],
                                      target_requirements: Dict[str, Any]) -> List[PatternEvolution]:
        """Analyze how architectural patterns should evolve"""
        logger.info(f"Analyzing pattern evolution from {current_patterns}")
        
        evolutions = self.pattern_evolution_engine.analyze_pattern_evolution(
            current_patterns, target_requirements
        )
        
        logger.info(f"Found {len(evolutions)} pattern evolution opportunities")
        return evolutions
    
    async def optimize_for_performance(self, current_metrics: PerformanceMetrics,
                                     target_metrics: PerformanceMetrics,
                                     architecture_options: List[ArchitecturalOption]) -> Dict[str, Any]:
        """Optimize architecture for performance requirements"""
        logger.info("Optimizing architecture for performance requirements")
        
        optimization_plan = self.performance_optimizer.optimize_for_performance(
            current_metrics, target_metrics, architecture_options
        )
        
        logger.info(f"Performance optimization plan created for {optimization_plan['recommended_architecture'].name}")
        return optimization_plan
    
    async def analyze_microservice_evolution(self, current_architecture: Dict[str, Any],
                                           requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze microservice architecture evolution"""
        logger.info("Analyzing microservice architecture evolution")
        
        analysis = self.microservice_analyzer.analyze_microservice_evolution(
            current_architecture, requirements
        )
        
        logger.info("Microservice evolution analysis completed")
        return analysis
    
    async def analyze_trade_offs(self, options: List[ArchitecturalOption],
                               context: DecisionContext) -> Dict[str, Any]:
        """Analyze trade-offs between architectural options"""
        logger.info(f"Analyzing trade-offs for {len(options)} architectural options")
        
        trade_off_analysis = {
            "option_comparisons": [],
            "criteria_analysis": {},
            "recommendation_rationale": [],
            "decision_matrix": {}
        }
        
        # Score all options for comparison
        option_scores = {}
        for option in options:
            scores = {}
            for criteria in DecisionCriteria:
                score = self._calculate_criteria_score(option, criteria, context)
                scores[criteria.value] = score
            option_scores[option.id] = scores
        
        # Generate pairwise comparisons
        for i, option1 in enumerate(options):
            for j, option2 in enumerate(options[i+1:], i+1):
                comparison = self._compare_options(option1, option2, option_scores, context)
                trade_off_analysis["option_comparisons"].append(comparison)
        
        # Analyze by criteria
        for criteria in DecisionCriteria:
            criteria_analysis = self._analyze_criteria_across_options(options, criteria, option_scores)
            trade_off_analysis["criteria_analysis"][criteria.value] = criteria_analysis
        
        # Generate decision matrix
        trade_off_analysis["decision_matrix"] = self._generate_decision_matrix(options, option_scores)
        
        # Generate recommendation rationale
        trade_off_analysis["recommendation_rationale"] = self._generate_recommendation_rationale(
            options, option_scores, context
        )
        
        logger.info("Trade-off analysis completed")
        return trade_off_analysis
    
    async def assess_implementation_risk(self, option: ArchitecturalOption,
                                       context: DecisionContext) -> Dict[str, float]:
        """Assess implementation risks for an architectural option"""
        logger.info(f"Assessing implementation risk for: {option.name}")
        
        risks = {}
        
        # Technical risks
        risks.update(self._assess_technical_risks(option, context))
        
        # Organizational risks
        risks.update(self._assess_organizational_risks(option, context))
        
        # Timeline risks
        risks.update(self._assess_timeline_risks(option, context))
        
        # Cost risks
        risks.update(self._assess_cost_risks(option, context))
        
        # Calculate overall risk score
        if risks:
            avg_risk = sum(risks.values()) / len(risks)
            risks["overall_risk"] = avg_risk
        
        logger.info(f"Risk assessment completed. Overall risk: {risks.get('overall_risk', 0):.2f}")
        return risks
    
    async def generate_implementation_plan(self, option: ArchitecturalOption,
                                         context: DecisionContext) -> List[str]:
        """Generate detailed implementation plan for architectural option"""
        logger.info(f"Generating implementation plan for: {option.name}")
        
        plan = [
            "1. Architecture Design and Planning Phase",
            "   - Create detailed architecture diagrams",
            "   - Define component interfaces and contracts",
            "   - Establish development and deployment standards",
            "   - Set up monitoring and observability strategy"
        ]
        
        # Add pattern-specific steps
        for pattern in option.patterns:
            pattern_steps = self._get_pattern_implementation_steps(pattern)
            plan.extend([f"   - {step}" for step in pattern_steps])
        
        plan.extend([
            "2. Infrastructure Preparation Phase",
            "   - Set up development and testing environments",
            "   - Implement CI/CD pipelines",
            "   - Configure monitoring and logging infrastructure",
            "   - Set up security and compliance frameworks"
        ])
        
        # Add technology-specific steps
        for technology in option.technologies:
            tech_steps = self._get_technology_implementation_steps(technology)
            plan.extend([f"   - {step}" for step in tech_steps])
        
        plan.extend([
            "3. Implementation Phase",
            "   - Implement core components incrementally",
            "   - Integrate with existing systems",
            "   - Implement testing strategy",
            "   - Conduct performance testing and optimization",
            "",
            "4. Deployment and Validation Phase",
            "   - Deploy to staging environment",
            "   - Conduct comprehensive testing",
            "   - Perform security and compliance validation",
            "   - Deploy to production with monitoring",
            "",
            "5. Post-Deployment Optimization",
            "   - Monitor performance and reliability",
            "   - Gather user feedback",
            "   - Optimize based on real-world usage",
            "   - Document lessons learned and best practices"
        ])
        
        logger.info(f"Implementation plan generated with {len(plan)} steps")
        return plan
    
    def get_decision_history(self, limit: Optional[int] = None) -> List[DecisionAnalysis]:
        """Get decision history with optional limit"""
        if limit:
            return self.decision_history[-limit:]
        return self.decision_history.copy()
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about decision making"""
        if not self.decision_history:
            return {"total_decisions": 0}
        
        confidence_scores = [decision.confidence_score for decision in self.decision_history]
        decision_types = [decision.decision_id.split('_')[0] for decision in self.decision_history]
        
        return {
            "total_decisions": len(self.decision_history),
            "average_confidence": statistics.mean(confidence_scores),
            "confidence_std": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
            "decision_types": dict(pd.Series(decision_types).value_counts()),
            "recent_decisions": len([d for d in self.decision_history 
                                   if d.created_at > datetime.now() - timedelta(days=30)])
        }
    
    # Private helper methods - consolidated from original implementation
    
    def _generate_cache_key(self, context: DecisionContext, options: List[ArchitecturalOption]) -> str:
        """Generate cache key for decision context and options"""
        context_str = f"{context.decision_type.value}_{context.description}_{len(options)}"
        options_str = "_".join([option.id for option in options])
        combined = f"{context_str}_{options_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _generate_comprehensive_analysis(self, context: DecisionContext,
                                             recommended_option: ArchitecturalOption,
                                             alternative_options: List[ArchitecturalOption],
                                             current_metrics: Optional[PerformanceMetrics],
                                             target_metrics: Optional[PerformanceMetrics]) -> DecisionAnalysis:
        """Generate comprehensive analysis for the decision"""
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(recommended_option, context)
        
        # Assess risks
        risk_assessment = await self.assess_implementation_risk(recommended_option, context)
        
        # Generate trade-off analysis
        all_options = [recommended_option] + alternative_options
        trade_off_analysis = await self.analyze_trade_offs(all_options, context)
        
        # Generate implementation plan
        implementation_plan = await self.generate_implementation_plan(recommended_option, context)
        
        # Generate analysis summary
        analysis_summary = self._generate_analysis_summary(
            recommended_option, alternative_options, risk_assessment, confidence_score
        )
        
        # Generate decision rationale
        decision_rationale = self._generate_decision_rationale(
            recommended_option, context, risk_assessment, confidence_score
        )
        
        # Define success metrics
        success_metrics = self._define_success_metrics(recommended_option, context)
        
        # Generate monitoring recommendations
        monitoring_recommendations = self._generate_monitoring_recommendations(recommended_option, context)
        
        return DecisionAnalysis(
            decision_id=context.decision_id,
            recommended_option=recommended_option,
            alternative_options=alternative_options,
            analysis_summary=analysis_summary,
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            trade_off_analysis=trade_off_analysis,
            implementation_plan=implementation_plan,
            success_metrics=success_metrics,
            monitoring_recommendations=monitoring_recommendations,
            decision_rationale=decision_rationale
        )
    
    def _calculate_confidence_score(self, option: ArchitecturalOption, context: DecisionContext) -> float:
        """Calculate confidence score for the decision"""
        base_confidence = 0.7
        
        # Boost confidence based on scoring
        if option.scores:
            avg_score = sum(option.scores.values()) / len(option.scores)
            confidence_boost = (avg_score - 50) / 100  # Normalize to -0.5 to 0.5
            base_confidence += confidence_boost
        
        # Reduce confidence based on risks
        risk_penalty = len(option.risk_factors) * 0.05
        base_confidence -= risk_penalty
        
        # Reduce confidence based on complexity
        complexity_penalty = len(option.technologies) * 0.02
        base_confidence -= complexity_penalty
        
        # Boost confidence based on team expertise
        if context.team_expertise:
            expertise_boost = 0.0
            for tech in option.technologies:
                if tech.lower() in context.team_expertise:
                    expertise_boost += context.team_expertise[tech.lower()] / 1000  # Scale expertise
            base_confidence += expertise_boost
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_criteria_score(self, option: ArchitecturalOption, 
                                criteria: DecisionCriteria, context: DecisionContext) -> float:
        """Calculate score for a specific criteria"""
        if criteria in option.scores:
            return option.scores[criteria]
        
        # Fallback scoring logic
        base_score = 50.0
        
        if criteria == DecisionCriteria.COST:
            if context.budget and option.estimated_cost > 0:
                cost_ratio = option.estimated_cost / context.budget
                base_score = max(0, 100 - (cost_ratio * 100))
        elif criteria == DecisionCriteria.COMPLEXITY:
            base_score = max(0, 100 - (len(option.technologies) * 10))
        elif criteria == DecisionCriteria.RISK:
            base_score = max(0, 100 - (len(option.risk_factors) * 15))
        
        return base_score
    
    def _compare_options(self, option1: ArchitecturalOption, option2: ArchitecturalOption,
                        option_scores: Dict[str, Dict[str, float]], context: DecisionContext) -> Dict[str, Any]:
        """Compare two architectural options"""
        scores1 = option_scores[option1.id]
        scores2 = option_scores[option2.id]
        
        comparison = {
            "option1": {"id": option1.id, "name": option1.name},
            "option2": {"id": option2.id, "name": option2.name},
            "criteria_comparisons": {},
            "overall_winner": None,
            "trade_offs": []
        }
        
        wins1 = 0
        wins2 = 0
        
        for criteria, score1 in scores1.items():
            score2 = scores2[criteria]
            
            if score1 > score2:
                winner = option1.name
                wins1 += 1
            elif score2 > score1:
                winner = option2.name
                wins2 += 1
            else:
                winner = "tie"
            
            comparison["criteria_comparisons"][criteria] = {
                "score1": score1,
                "score2": score2,
                "winner": winner,
                "difference": abs(score1 - score2)
            }
        
        comparison["overall_winner"] = option1.name if wins1 > wins2 else option2.name if wins2 > wins1 else "tie"
        
        # Identify trade-offs
        comparison["trade_offs"] = self._identify_trade_offs(option1, option2, scores1, scores2)
        
        return comparison
    
    def _identify_trade_offs(self, option1: ArchitecturalOption, option2: ArchitecturalOption,
                           scores1: Dict[str, float], scores2: Dict[str, float]) -> List[str]:
        """Identify key trade-offs between options"""
        trade_offs = []
        
        for criteria, score1 in scores1.items():
            score2 = scores2[criteria]
            difference = abs(score1 - score2)
            
            if difference > 20:  # Significant difference
                if score1 > score2:
                    trade_offs.append(f"{option1.name} is significantly better at {criteria} than {option2.name}")
                else:
                    trade_offs.append(f"{option2.name} is significantly better at {criteria} than {option1.name}")
        
        return trade_offs
    
    def _analyze_criteria_across_options(self, options: List[ArchitecturalOption],
                                       criteria: DecisionCriteria,
                                       option_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze how all options perform on a specific criteria"""
        criteria_key = criteria.value
        scores = []
        option_names = []
        
        for option in options:
            if option.id in option_scores and criteria_key in option_scores[option.id]:
                scores.append(option_scores[option.id][criteria_key])
                option_names.append(option.name)
        
        if not scores:
            return {"error": f"No scores available for {criteria_key}"}
        
        best_idx = scores.index(max(scores))
        worst_idx = scores.index(min(scores))
        
        return {
            "criteria": criteria_key,
            "best_option": {"name": option_names[best_idx], "score": scores[best_idx]},
            "worst_option": {"name": option_names[worst_idx], "score": scores[worst_idx]},
            "average_score": statistics.mean(scores),
            "score_range": max(scores) - min(scores),
            "all_scores": dict(zip(option_names, scores))
        }
    
    def _generate_decision_matrix(self, options: List[ArchitecturalOption],
                                option_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate decision matrix for options and criteria"""
        matrix = {}
        criteria_list = []
        
        # Get all criteria
        for option in options:
            if option.id in option_scores:
                criteria_list.extend(option_scores[option.id].keys())
        
        criteria_list = list(set(criteria_list))
        
        # Build matrix
        for option in options:
            option_scores_dict = option_scores.get(option.id, {})
            matrix[option.name] = {criteria: option_scores_dict.get(criteria, 0) for criteria in criteria_list}
        
        return {
            "matrix": matrix,
            "criteria": criteria_list,
            "options": [option.name for option in options]
        }
    
    def _generate_recommendation_rationale(self, options: List[ArchitecturalOption],
                                         option_scores: Dict[str, Dict[str, float]],
                                         context: DecisionContext) -> List[str]:
        """Generate rationale for the recommendation"""
        if not options:
            return ["No options provided for analysis"]
        
        best_option = options[0]  # Assuming first option is the recommended one
        best_scores = option_scores.get(best_option.id, {})
        
        rationale = [
            f"Recommended {best_option.name} based on comprehensive multi-criteria analysis"
        ]
        
        # Identify strongest criteria
        if best_scores:
            sorted_criteria = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
            top_criteria = sorted_criteria[:3]
            
            for criteria, score in top_criteria:
                if score > 70:
                    rationale.append(f"Excels in {criteria} with score of {score:.1f}")
        
        # Consider context factors
        if context.priority == DecisionPriority.CRITICAL:
            rationale.append("High priority decision requiring reliable and proven solution")
        
        if context.budget and best_option.estimated_cost <= context.budget:
            rationale.append(f"Fits within budget constraint of ${context.budget:,.2f}")
        
        if context.timeline:
            rationale.append(f"Can be implemented within timeline constraint")
        
        return rationale
    
    def _assess_technical_risks(self, option: ArchitecturalOption, context: DecisionContext) -> Dict[str, float]:
        """Assess technical risks"""
        risks = {}
        
        # Technology maturity risk
        new_tech_count = len([tech for tech in option.technologies if 'new' in tech.lower() or 'experimental' in tech.lower()])
        if new_tech_count > 0:
            risks["technology_maturity"] = min(0.8, new_tech_count * 0.3)
        
        # Complexity risk
        if len(option.technologies) > 5:
            risks["technology_complexity"] = min(0.9, (len(option.technologies) - 5) * 0.2)
        
        # Pattern complexity risk
        complex_patterns = {ArchitecturalPattern.MICROSERVICES, ArchitecturalPattern.EVENT_DRIVEN, ArchitecturalPattern.CQRS}
        if any(pattern in complex_patterns for pattern in option.patterns):
            risks["architectural_complexity"] = 0.6
        
        # Technical debt risk
        if option.technical_debt > 0.5:
            risks["technical_debt"] = option.technical_debt
        
        return risks
    
    def _assess_organizational_risks(self, option: ArchitecturalOption, context: DecisionContext) -> Dict[str, float]:
        """Assess organizational risks"""
        risks = {}
        
        # Team expertise risk
        if context.team_expertise:
            expertise_gap = 0.0
            for tech in option.technologies:
                tech_expertise = context.team_expertise.get(tech.lower(), 0)
                if tech_expertise < 50:  # Low expertise
                    expertise_gap += (50 - tech_expertise) / 100
            
            if expertise_gap > 0:
                risks["team_expertise_gap"] = min(0.9, expertise_gap)
        
        # Team size risk
        if context.team_size < 5 and any(pattern == ArchitecturalPattern.MICROSERVICES for pattern in option.patterns):
            risks["team_size_inadequate"] = 0.7
        
        # Change management risk
        if len(option.patterns) > 2:
            risks["change_management"] = 0.5
        
        return risks
    
    def _assess_timeline_risks(self, option: ArchitecturalOption, context: DecisionContext) -> Dict[str, float]:
        """Assess timeline risks"""
        risks = {}
        
        # Implementation time risk
        if option.implementation_time > 90:  # More than 3 months
            risks["long_implementation"] = min(0.8, (option.implementation_time - 90) / 180)
        
        # Effort estimation risk
        if option.estimated_effort > 2000:  # High effort
            risks["effort_underestimation"] = 0.6
        
        # Timeline pressure risk
        if context.timeline and context.timeline < datetime.now() + timedelta(days=option.implementation_time):
            risks["timeline_pressure"] = 0.8
        
        return risks
    
    def _assess_cost_risks(self, option: ArchitecturalOption, context: DecisionContext) -> Dict[str, float]:
        """Assess cost risks"""
        risks = {}
        
        # Budget overrun risk
        if context.budget and option.estimated_cost > context.budget:
            overrun_ratio = option.estimated_cost / context.budget
            risks["budget_overrun"] = min(0.9, (overrun_ratio - 1.0))
        
        # Hidden cost risk
        if len(option.technologies) > 3:
            risks["hidden_costs"] = 0.4
        
        # Operational cost risk
        for pattern in option.patterns:
            if pattern == ArchitecturalPattern.MICROSERVICES:
                risks["operational_cost_increase"] = 0.5
            elif pattern == ArchitecturalPattern.SERVERLESS:
                risks["usage_based_cost_variability"] = 0.6
        
        return risks
    
    def _get_pattern_implementation_steps(self, pattern: ArchitecturalPattern) -> List[str]:
        """Get implementation steps specific to an architectural pattern"""
        steps_map = {
            ArchitecturalPattern.MICROSERVICES: [
                "Define service boundaries using Domain-Driven Design",
                "Implement service discovery mechanism",
                "Set up inter-service communication protocols",
                "Implement distributed data management",
                "Set up monitoring and distributed tracing"
            ],
            ArchitecturalPattern.EVENT_DRIVEN: [
                "Design event schemas and contracts",
                "Implement event bus or message broker",
                "Create event handlers and processors",
                "Implement event sourcing if needed",
                "Set up event monitoring and replay capabilities"
            ],
            ArchitecturalPattern.HEXAGONAL: [
                "Define core business logic boundaries",
                "Implement adapter interfaces",
                "Create concrete adapter implementations",
                "Set up dependency injection",
                "Implement comprehensive testing strategy"
            ],
            ArchitecturalPattern.SERVERLESS: [
                "Break down functionality into discrete functions",
                "Implement function deployment packages",
                "Set up event triggers and schedulers",
                "Implement state management strategy",
                "Configure monitoring and logging"
            ]
        }
        
        return steps_map.get(pattern, [f"Implement {pattern.value} architectural pattern"])
    
    def _get_technology_implementation_steps(self, technology: str) -> List[str]:
        """Get implementation steps specific to a technology"""
        # This could be expanded with specific technology knowledge
        return [f"Set up and configure {technology}", f"Integrate {technology} with existing systems"]
    
    def _generate_analysis_summary(self, recommended_option: ArchitecturalOption,
                                 alternative_options: List[ArchitecturalOption],
                                 risk_assessment: Dict[str, float],
                                 confidence_score: float) -> str:
        """Generate analysis summary"""
        summary_parts = [
            f"Recommended architectural option: {recommended_option.name}",
            f"Decision confidence: {confidence_score:.1%}",
            f"Estimated implementation time: {recommended_option.implementation_time} days",
            f"Estimated cost: ${recommended_option.estimated_cost:,.2f}"
        ]
        
        if risk_assessment:
            max_risk = max(risk_assessment.values()) if risk_assessment else 0
            summary_parts.append(f"Maximum identified risk level: {max_risk:.1%}")
        
        if alternative_options:
            alt_names = [opt.name for opt in alternative_options[:2]]
            summary_parts.append(f"Alternative options considered: {', '.join(alt_names)}")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_decision_rationale(self, option: ArchitecturalOption, context: DecisionContext,
                                   risk_assessment: Dict[str, float], confidence_score: float) -> str:
        """Generate detailed decision rationale"""
        rationale_parts = [
            f"Selected {option.name} as the optimal architectural solution for {context.description}."
        ]
        
        # Add strength-based rationale
        if option.scores:
            strong_areas = [criteria.replace('_', ' ').title() 
                          for criteria, score in option.scores.items() if score > 75]
            if strong_areas:
                rationale_parts.append(f"This option excels in: {', '.join(strong_areas)}.")
        
        # Add context-based rationale
        rationale_parts.append(f"The decision aligns with the {context.priority.value} priority level of this decision.")
        
        if context.budget and option.estimated_cost <= context.budget:
            rationale_parts.append(f"The solution fits within the allocated budget of ${context.budget:,.2f}.")
        
        # Add risk consideration
        if risk_assessment:
            avg_risk = sum(risk_assessment.values()) / len(risk_assessment)
            if avg_risk < 0.3:
                rationale_parts.append("Risk analysis indicates this is a low-risk implementation.")
            elif avg_risk < 0.6:
                rationale_parts.append("Risk analysis indicates manageable risks with proper mitigation.")
            else:
                rationale_parts.append("While there are significant risks, the benefits justify the decision with careful risk management.")
        
        # Add confidence statement
        if confidence_score > 0.8:
            rationale_parts.append("High confidence in this recommendation based on comprehensive analysis.")
        elif confidence_score > 0.6:
            rationale_parts.append("Moderate confidence with recommendation to validate key assumptions.")
        else:
            rationale_parts.append("Lower confidence suggests need for additional analysis or prototyping.")
        
        return " ".join(rationale_parts)
    
    def _define_success_metrics(self, option: ArchitecturalOption, context: DecisionContext) -> List[str]:
        """Define success metrics for the architectural decision"""
        metrics = [
            "Implementation completed within estimated timeline",
            "Budget adherence within 10% of estimates",
            "All functional requirements successfully implemented",
            "Performance targets met or exceeded",
            "No critical security vulnerabilities introduced"
        ]
        
        # Add pattern-specific metrics
        for pattern in option.patterns:
            if pattern == ArchitecturalPattern.MICROSERVICES:
                metrics.extend([
                    "Services can be deployed independently",
                    "Average service response time < 200ms",
                    "Service availability > 99.9%"
                ])
            elif pattern == ArchitecturalPattern.EVENT_DRIVEN:
                metrics.extend([
                    "Event processing latency < 100ms",
                    "Event delivery reliability > 99.95%",
                    "System handles expected event volume without degradation"
                ])
        
        # Add context-specific metrics
        if context.decision_type == DecisionType.PERFORMANCE_OPTIMIZATION:
            metrics.append("Performance improvement of at least 20% measured")
        elif context.decision_type == DecisionType.SCALING_STRATEGY:
            metrics.append("System successfully scales to handle 10x current load")
        
        return metrics[:10]  # Limit to top 10 metrics
    
    def _generate_monitoring_recommendations(self, option: ArchitecturalOption, context: DecisionContext) -> List[str]:
        """Generate monitoring recommendations for the architectural solution"""
        recommendations = [
            "Implement comprehensive application performance monitoring (APM)",
            "Set up infrastructure monitoring for all components",
            "Establish logging aggregation and analysis",
            "Create dashboards for key business and technical metrics",
            "Implement alerting for critical system events"
        ]
        
        # Add pattern-specific monitoring
        for pattern in option.patterns:
            if pattern == ArchitecturalPattern.MICROSERVICES:
                recommendations.extend([
                    "Implement distributed tracing across services",
                    "Monitor service-to-service communication patterns",
                    "Track service dependency health and latency"
                ])
            elif pattern == ArchitecturalPattern.EVENT_DRIVEN:
                recommendations.extend([
                    "Monitor event queue depths and processing rates",
                    "Track event processing latency and failures",
                    "Implement event flow visualization"
                ])
        
        # Add technology-specific monitoring
        for technology in option.technologies:
            if 'database' in technology.lower():
                recommendations.append(f"Monitor {technology} performance and query optimization")
            elif 'cache' in technology.lower():
                recommendations.append(f"Monitor {technology} hit rates and memory usage")
        
        return list(set(recommendations))[:8]  # Remove duplicates and limit


__all__ = ['ArchitecturalDecisionEngine']