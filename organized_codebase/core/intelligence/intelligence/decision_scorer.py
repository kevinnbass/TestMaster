"""
Architectural Decision Scoring System
====================================

Modularized from architectural_decision_engine.py for better maintainability.
Provides multi-criteria scoring and evaluation of architectural options.

Author: Agent E - Infrastructure Consolidation
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

from .data_models import (
    DecisionCriteria, ArchitecturalPattern, ArchitecturalOption, DecisionContext
)

logger = logging.getLogger(__name__)


class DecisionScorer:
    """Scores architectural options against criteria"""
    
    def __init__(self):
        self.weights = {
            DecisionCriteria.PERFORMANCE: 0.15,
            DecisionCriteria.SCALABILITY: 0.15,
            DecisionCriteria.MAINTAINABILITY: 0.15,
            DecisionCriteria.SECURITY: 0.15,
            DecisionCriteria.COST: 0.10,
            DecisionCriteria.COMPLEXITY: 0.10,
            DecisionCriteria.TIME_TO_MARKET: 0.05,
            DecisionCriteria.TEAM_EXPERTISE: 0.05,
            DecisionCriteria.RISK: 0.10
        }
    
    def calculate_weighted_score(self, scores: Dict[DecisionCriteria, float]) -> float:
        """Calculate weighted score for an option"""
        total_score = 0.0
        total_weight = 0.0
        
        for criteria, score in scores.items():
            if criteria in self.weights:
                weight = self.weights[criteria]
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def score_option(self, option: ArchitecturalOption, context: DecisionContext) -> float:
        """Score a single architectural option"""
        scores = {}
        
        # Performance scoring
        scores[DecisionCriteria.PERFORMANCE] = self._score_performance(option, context)
        
        # Scalability scoring
        scores[DecisionCriteria.SCALABILITY] = self._score_scalability(option, context)
        
        # Maintainability scoring
        scores[DecisionCriteria.MAINTAINABILITY] = self._score_maintainability(option, context)
        
        # Security scoring
        scores[DecisionCriteria.SECURITY] = self._score_security(option, context)
        
        # Cost scoring
        scores[DecisionCriteria.COST] = self._score_cost(option, context)
        
        # Complexity scoring
        scores[DecisionCriteria.COMPLEXITY] = self._score_complexity(option, context)
        
        # Risk scoring
        scores[DecisionCriteria.RISK] = self._score_risk(option, context)
        
        option.scores = scores
        return self.calculate_weighted_score(scores)
    
    def _score_performance(self, option: ArchitecturalOption, context: DecisionContext) -> float:
        """Score performance characteristics"""
        base_score = 50.0
        
        # Boost for high-performance patterns
        high_perf_patterns = {ArchitecturalPattern.MICROSERVICES, ArchitecturalPattern.EVENT_DRIVEN}
        if any(pattern in high_perf_patterns for pattern in option.patterns):
            base_score += 20.0
        
        # Consider technical debt impact
        base_score -= min(option.technical_debt * 10, 30.0)
        
        return max(0.0, min(100.0, base_score))
    
    def _score_scalability(self, option: ArchitecturalOption, context: DecisionContext) -> float:
        """Score scalability characteristics"""
        base_score = 50.0
        
        # Boost for scalable patterns
        scalable_patterns = {ArchitecturalPattern.MICROSERVICES, ArchitecturalPattern.SERVERLESS}
        if any(pattern in scalable_patterns for pattern in option.patterns):
            base_score += 25.0
        
        # Consider implementation effort
        if option.estimated_effort > 1000:  # High effort might indicate complex scaling
            base_score -= 15.0
        
        return max(0.0, min(100.0, base_score))
    
    def _score_maintainability(self, option: ArchitecturalOption, context: DecisionContext) -> float:
        """Score maintainability characteristics"""
        base_score = 50.0
        
        # Boost for maintainable patterns
        maintainable_patterns = {ArchitecturalPattern.MODULAR_MONOLITH, ArchitecturalPattern.HEXAGONAL}
        if any(pattern in maintainable_patterns for pattern in option.patterns):
            base_score += 20.0
        
        # Penalize high technical debt
        base_score -= option.technical_debt * 20
        
        return max(0.0, min(100.0, base_score))
    
    def _score_security(self, option: ArchitecturalOption, context: DecisionContext) -> float:
        """Score security characteristics"""
        base_score = 60.0
        
        # Consider security-focused patterns
        secure_patterns = {ArchitecturalPattern.HEXAGONAL, ArchitecturalPattern.API_GATEWAY}
        if any(pattern in secure_patterns for pattern in option.patterns):
            base_score += 15.0
        
        # Consider risk factors
        security_risks = [risk for risk in option.risk_factors if 'security' in risk.lower()]
        base_score -= len(security_risks) * 10
        
        return max(0.0, min(100.0, base_score))
    
    def _score_cost(self, option: ArchitecturalOption, context: DecisionContext) -> float:
        """Score cost effectiveness"""
        if not context.budget or option.estimated_cost == 0:
            return 50.0
        
        cost_ratio = option.estimated_cost / context.budget
        if cost_ratio <= 0.5:
            return 90.0
        elif cost_ratio <= 0.8:
            return 70.0
        elif cost_ratio <= 1.0:
            return 50.0
        else:
            return max(0.0, 50.0 - (cost_ratio - 1.0) * 100)
    
    def _score_complexity(self, option: ArchitecturalOption, context: DecisionContext) -> float:
        """Score complexity (lower complexity = higher score)"""
        base_score = 80.0
        
        # Penalize complex patterns for small teams
        complex_patterns = {ArchitecturalPattern.MICROSERVICES, ArchitecturalPattern.SERVICE_MESH}
        if any(pattern in complex_patterns for pattern in option.patterns):
            if context.team_size < 10:
                base_score -= 30.0
            else:
                base_score -= 15.0
        
        # Consider number of technologies
        base_score -= len(option.technologies) * 2
        
        return max(0.0, min(100.0, base_score))
    
    def _score_risk(self, option: ArchitecturalOption, context: DecisionContext) -> float:
        """Score risk level (lower risk = higher score)"""
        base_score = 70.0
        
        # Penalize based on risk factors
        base_score -= len(option.risk_factors) * 8
        
        # Consider team expertise
        tech_expertise = 0.0
        for tech in option.technologies:
            if tech.lower() in context.team_expertise:
                tech_expertise += context.team_expertise[tech.lower()]
        
        if option.technologies:
            avg_expertise = tech_expertise / len(option.technologies)
            base_score += (avg_expertise - 50) * 0.4  # Scale expertise impact
        
        return max(0.0, min(100.0, base_score))


__all__ = ['DecisionScorer']