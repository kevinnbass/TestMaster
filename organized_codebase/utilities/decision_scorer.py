"""
Pattern Intelligence Decision Scorer
===================================

Multi-criteria decision analysis with weighted scoring for architectural decisions.
Extracted from architectural_decision_engine.py for enterprise modular architecture.

Agent D Implementation - Hour 12-13: Revolutionary Intelligence Modularization
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.data_models import (
    DecisionCriteria, ArchitecturalOption, DecisionContext, 
    ArchitecturalPattern, DecisionAnalysis, DecisionType
)


@dataclass
class CriteriaWeights:
    """Weighted criteria for decision scoring"""
    performance: float = 0.15
    scalability: float = 0.15
    maintainability: float = 0.15
    security: float = 0.15
    cost: float = 0.10
    complexity: float = 0.10
    flexibility: float = 0.05
    reliability: float = 0.05
    team_expertise: float = 0.05
    time_to_market: float = 0.025
    vendor_lock_in: float = 0.025
    future_proofing: float = 0.025
    
    def normalize(self) -> 'CriteriaWeights':
        """Normalize weights to sum to 1.0"""
        total = (self.performance + self.scalability + self.maintainability + 
                self.security + self.cost + self.complexity + self.flexibility +
                self.reliability + self.team_expertise + self.time_to_market +
                self.vendor_lock_in + self.future_proofing)
        
        if total == 0:
            return self
        
        return CriteriaWeights(
            performance=self.performance / total,
            scalability=self.scalability / total,
            maintainability=self.maintainability / total,
            security=self.security / total,
            cost=self.cost / total,
            complexity=self.complexity / total,
            flexibility=self.flexibility / total,
            reliability=self.reliability / total,
            team_expertise=self.team_expertise / total,
            time_to_market=self.time_to_market / total,
            vendor_lock_in=self.vendor_lock_in / total,
            future_proofing=self.future_proofing / total
        )


class DecisionScorer:
    """Advanced multi-criteria decision analysis scorer"""
    
    def __init__(self, custom_weights: CriteriaWeights = None):
        self.weights = custom_weights.normalize() if custom_weights else CriteriaWeights().normalize()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Pattern-specific performance characteristics
        self.pattern_characteristics = {
            ArchitecturalPattern.MICROSERVICES: {
                'performance': 0.7, 'scalability': 0.9, 'maintainability': 0.6,
                'security': 0.8, 'cost': 0.4, 'complexity': 0.3
            },
            ArchitecturalPattern.MONOLITH: {
                'performance': 0.8, 'scalability': 0.4, 'maintainability': 0.7,
                'security': 0.7, 'cost': 0.8, 'complexity': 0.8
            },
            ArchitecturalPattern.MODULAR_MONOLITH: {
                'performance': 0.8, 'scalability': 0.6, 'maintainability': 0.8,
                'security': 0.7, 'cost': 0.7, 'complexity': 0.7
            },
            ArchitecturalPattern.LAYERED: {
                'performance': 0.6, 'scalability': 0.5, 'maintainability': 0.8,
                'security': 0.8, 'cost': 0.8, 'complexity': 0.8
            },
            ArchitecturalPattern.HEXAGONAL: {
                'performance': 0.7, 'scalability': 0.7, 'maintainability': 0.9,
                'security': 0.8, 'cost': 0.6, 'complexity': 0.6
            },
            ArchitecturalPattern.EVENT_DRIVEN: {
                'performance': 0.8, 'scalability': 0.9, 'maintainability': 0.6,
                'security': 0.7, 'cost': 0.5, 'complexity': 0.4
            },
            ArchitecturalPattern.SERVERLESS: {
                'performance': 0.6, 'scalability': 0.9, 'maintainability': 0.7,
                'security': 0.8, 'cost': 0.7, 'complexity': 0.5
            }
        }
    
    def score_options(self, options: List[ArchitecturalOption], 
                     context: DecisionContext) -> Dict[str, Dict[DecisionCriteria, float]]:
        """Score all options against all criteria"""
        scores = {}
        
        for option in options:
            option_scores = {}
            
            for criteria in DecisionCriteria:
                score = self._score_option_for_criteria(option, criteria, context)
                option_scores[criteria] = score
            
            scores[option.option_id] = option_scores
            
            self.logger.debug(f"Scored option {option.name}: average={sum(option_scores.values())/len(option_scores):.2f}")
        
        return scores
    
    def _score_option_for_criteria(self, option: ArchitecturalOption, 
                                  criteria: DecisionCriteria,
                                  context: DecisionContext) -> float:
        """Score a single option for a specific criteria"""
        try:
            # Base score from pattern characteristics
            base_score = self.pattern_characteristics.get(option.pattern, {}).get(criteria.value, 0.5)
            
            # Apply context-specific adjustments
            adjusted_score = self._apply_context_adjustments(base_score, option, criteria, context)
            
            # Apply option-specific characteristics
            final_score = self._apply_option_characteristics(adjusted_score, option, criteria)
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error scoring option {option.name} for {criteria.value}: {e}")
            return 0.5  # Default neutral score
    
    def _apply_context_adjustments(self, base_score: float, option: ArchitecturalOption,
                                 criteria: DecisionCriteria, context: DecisionContext) -> float:
        """Apply context-specific adjustments to base score"""
        score = base_score
        
        try:
            if criteria == DecisionCriteria.TEAM_EXPERTISE:
                # Boost score if team has expertise in required skills
                skill_overlap = len(set(option.required_skills) & set(context.team_expertise_areas))
                if option.required_skills:
                    expertise_ratio = skill_overlap / len(option.required_skills)
                    score = base_score * (0.5 + 0.5 * expertise_ratio)
            
            elif criteria == DecisionCriteria.COST:
                # Adjust based on budget constraints
                if option.estimated_cost <= context.budget_constraints:
                    score = base_score * 1.2  # Boost if within budget
                else:
                    over_budget_ratio = option.estimated_cost / context.budget_constraints
                    score = base_score * (1.0 / over_budget_ratio)  # Penalize if over budget
            
            elif criteria == DecisionCriteria.TIME_TO_MARKET:
                # Adjust based on timeline constraints
                if option.implementation_time <= context.timeline_constraints:
                    score = base_score * 1.2  # Boost if within timeline
                else:
                    time_ratio = option.implementation_time.total_seconds() / context.timeline_constraints.total_seconds()
                    score = base_score * (1.0 / time_ratio)  # Penalize if over timeline
            
            elif criteria == DecisionCriteria.COMPLEXITY:
                # Adjust based on team size and risk tolerance
                complexity_penalty = option.estimated_complexity
                if context.team_size < 5:
                    complexity_penalty *= 1.3  # Small teams penalized more for complexity
                if context.risk_tolerance == "low":
                    complexity_penalty *= 1.2  # Risk-averse teams penalized more
                
                score = base_score * (1.0 - complexity_penalty * 0.3)
            
            elif criteria == DecisionCriteria.SCALABILITY:
                # Boost based on scalability requirements
                if context.scalability_requirements:
                    max_scale = max(context.scalability_requirements.values())
                    if max_scale > 1000:  # High scalability needs
                        if option.pattern in [ArchitecturalPattern.MICROSERVICES, 
                                            ArchitecturalPattern.EVENT_DRIVEN,
                                            ArchitecturalPattern.SERVERLESS]:
                            score = base_score * 1.3
            
            elif criteria == DecisionCriteria.SECURITY:
                # Boost if security requirements are high
                if len(context.security_requirements) > 3:
                    if option.pattern in [ArchitecturalPattern.HEXAGONAL,
                                        ArchitecturalPattern.LAYERED]:
                        score = base_score * 1.2
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error applying context adjustments: {e}")
            return base_score
    
    def _apply_option_characteristics(self, score: float, option: ArchitecturalOption,
                                    criteria: DecisionCriteria) -> float:
        """Apply option-specific characteristics"""
        try:
            # Risk factor penalties
            if criteria in [DecisionCriteria.RELIABILITY, DecisionCriteria.MAINTAINABILITY]:
                risk_penalty = len(option.risk_factors) * 0.05
                score = score * (1.0 - risk_penalty)
            
            # Dependency complexity penalties
            if criteria == DecisionCriteria.COMPLEXITY:
                dependency_penalty = len(option.dependencies) * 0.03
                score = score * (1.0 - dependency_penalty)
            
            # Advantage boosts
            if criteria == DecisionCriteria.FLEXIBILITY and "flexible" in str(option.advantages).lower():
                score = score * 1.1
            
            if criteria == DecisionCriteria.PERFORMANCE and "performance" in str(option.advantages).lower():
                score = score * 1.1
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error applying option characteristics: {e}")
            return score
    
    def calculate_weighted_scores(self, criteria_scores: Dict[str, Dict[DecisionCriteria, float]]) -> Dict[str, float]:
        """Calculate weighted scores for all options"""
        weighted_scores = {}
        
        for option_id, scores in criteria_scores.items():
            weighted_score = 0.0
            
            for criteria, score in scores.items():
                weight = getattr(self.weights, criteria.value, 0.0)
                weighted_score += score * weight
            
            weighted_scores[option_id] = weighted_score
            
            self.logger.debug(f"Weighted score for {option_id}: {weighted_score:.3f}")
        
        return weighted_scores
    
    def generate_recommendations(self, options: List[ArchitecturalOption], 
                               criteria_scores: Dict[str, Dict[DecisionCriteria, float]],
                               weighted_scores: Dict[str, float],
                               context: DecisionContext) -> Tuple[str, float, str]:
        """Generate recommendations based on scores"""
        # Find best option
        best_option_id = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
        best_score = weighted_scores[best_option_id]
        
        # Find the option object
        best_option = next(opt for opt in options if opt.option_id == best_option_id)
        
        # Calculate confidence based on score gap and consistency
        confidence = self._calculate_confidence(weighted_scores, criteria_scores, best_option_id)
        
        # Generate rationale
        rationale = self._generate_rationale(best_option, criteria_scores[best_option_id], context)
        
        self.logger.info(f"Recommended option: {best_option.name} (score: {best_score:.3f}, confidence: {confidence:.3f})")
        
        return best_option_id, confidence, rationale
    
    def _calculate_confidence(self, weighted_scores: Dict[str, float],
                            criteria_scores: Dict[str, Dict[DecisionCriteria, float]],
                            best_option_id: str) -> float:
        """Calculate confidence in the recommendation"""
        try:
            scores_list = list(weighted_scores.values())
            best_score = weighted_scores[best_option_id]
            
            if len(scores_list) < 2:
                return 0.7  # Default confidence for single option
            
            # Sort scores in descending order
            sorted_scores = sorted(scores_list, reverse=True)
            second_best = sorted_scores[1]
            
            # Score gap factor
            score_gap = best_score - second_best
            gap_factor = min(1.0, score_gap * 2)  # Normalize gap
            
            # Consistency factor (how consistently the best option scores well)
            best_criteria_scores = list(criteria_scores[best_option_id].values())
            consistency = 1.0 - (max(best_criteria_scores) - min(best_criteria_scores))
            consistency_factor = max(0.0, consistency)
            
            # Overall confidence
            confidence = (gap_factor * 0.6 + consistency_factor * 0.4)
            
            return max(0.3, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _generate_rationale(self, option: ArchitecturalOption,
                          criteria_scores: Dict[DecisionCriteria, float],
                          context: DecisionContext) -> str:
        """Generate human-readable rationale for the recommendation"""
        try:
            # Find top strengths
            sorted_criteria = sorted(criteria_scores.items(), key=lambda x: x[1], reverse=True)
            top_strengths = [criteria.value for criteria, score in sorted_criteria[:3] if score > 0.7]
            
            # Find weaknesses
            weaknesses = [criteria.value for criteria, score in sorted_criteria if score < 0.4]
            
            rationale_parts = [
                f"Recommended {option.pattern.value} pattern based on analysis of {len(criteria_scores)} criteria."
            ]
            
            if top_strengths:
                rationale_parts.append(f"Key strengths: {', '.join(top_strengths)}.")
            
            if option.estimated_cost <= context.budget_constraints:
                rationale_parts.append("Within budget constraints.")
            
            if option.implementation_time <= context.timeline_constraints:
                rationale_parts.append("Meets timeline requirements.")
            
            if weaknesses:
                rationale_parts.append(f"Areas requiring attention: {', '.join(weaknesses)}.")
            
            return " ".join(rationale_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating rationale: {e}")
            return f"Recommended {option.pattern.value} pattern based on multi-criteria analysis."
    
    def analyze_trade_offs(self, options: List[ArchitecturalOption],
                          criteria_scores: Dict[str, Dict[DecisionCriteria, float]]) -> Dict[str, List[str]]:
        """Analyze trade-offs between different criteria"""
        trade_offs = {}
        
        try:
            for criteria in DecisionCriteria:
                criteria_trade_offs = []
                
                # Find options that score well on this criteria
                good_options = [
                    option for option in options
                    if criteria_scores[option.option_id][criteria] > 0.7
                ]
                
                # Find what they sacrifice
                for option in good_options:
                    weak_areas = [
                        other_criteria.value for other_criteria, score in criteria_scores[option.option_id].items()
                        if score < 0.4 and other_criteria != criteria
                    ]
                    
                    if weak_areas:
                        trade_offs[criteria.value] = weak_areas
                
                trade_offs[criteria.value] = list(set(trade_offs.get(criteria.value, [])))
            
            return trade_offs
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade-offs: {e}")
            return {}


def create_decision_scorer(custom_weights: CriteriaWeights = None) -> DecisionScorer:
    """Factory function to create decision scorer"""
    return DecisionScorer(custom_weights)