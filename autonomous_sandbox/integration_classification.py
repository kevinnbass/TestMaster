#!/usr/bin/env python3
"""
Intelligence Integration Engine Classification Module
=====================================================

Classification logic and methods for the intelligence integration system.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass

from integration_models import IntegrationMethod, ConfidenceFactors, ClassificationResult


class IntelligenceClassificationEngine:
    """Handles classification logic for intelligence integration"""

    def __init__(self, config: Dict[str, Any], classification_taxonomy: Dict[str, Any]):
        """Initialize the classification engine"""
        self.config = config
        self.classification_taxonomy = classification_taxonomy

    def determine_integrated_classification(self, integration_method: IntegrationMethod,
                                           llm_entry, static_analysis, confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """Determine the integrated classification using the configured method"""

        if integration_method == IntegrationMethod.LLM_DOMINANT:
            return self._llm_dominant_classification(llm_entry, static_analysis, confidence_factors)

        elif integration_method == IntegrationMethod.STATIC_DOMINANT:
            return self._static_dominant_classification(llm_entry, static_analysis, confidence_factors)

        elif integration_method == IntegrationMethod.CONSENSUS_WITH_FALLBACK:
            return self._consensus_classification(llm_entry, static_analysis, confidence_factors)

        elif integration_method == IntegrationMethod.ADAPTIVE_CONFIDENCE:
            return self._adaptive_confidence_classification(llm_entry, static_analysis, confidence_factors)

        else:  # WEIGHTED_VOTING (default)
            return self._weighted_voting_classification(llm_entry, static_analysis, confidence_factors)

    def _weighted_voting_classification(self, llm_entry, static_analysis,
                                      confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """Use weighted voting to determine classification"""

        # Collect votes from different sources
        votes = defaultdict(float)

        # LLM vote
        votes[llm_entry.primary_classification] += confidence_factors.llm_confidence * 0.35

        # Semantic analysis vote
        if static_analysis.semantic:
            semantic_purpose = static_analysis.semantic.get('primary_purpose', 'unknown')
            mapped_category = self.classification_taxonomy['category_mappings'].get(semantic_purpose, semantic_purpose)
            votes[mapped_category] += confidence_factors.semantic_confidence * 0.20

        # Pattern analysis vote
        if static_analysis.pattern:
            pattern_votes = self._get_pattern_based_classification(static_analysis.pattern)
            for category, confidence in pattern_votes.items():
                votes[category] += confidence * 0.15

        # Quality analysis vote (lower weight)
        if static_analysis.quality:
            quality_category = self._get_quality_based_classification(static_analysis.quality)
            votes[quality_category] += confidence_factors.quality_confidence * 0.10

        # Determine winner
        if votes:
            winner = max(votes.items(), key=lambda x: x[1])
            primary_classification = winner[0]
            confidence = min(1.0, winner[1])

            # Get secondary classifications
            threshold = winner[1] * 0.6
            secondary_classifications = [cat for cat, score in votes.items()
                                       if score >= threshold and cat != primary_classification]

            # Alternative classifications
            alternative_classifications = sorted(votes.items(), key=lambda x: x[1], reverse=True)[1:4]
        else:
            primary_classification = 'uncategorized'
            confidence = 0.5
            secondary_classifications = []
            alternative_classifications = []

        return ClassificationResult(
            primary_classification=primary_classification,
            secondary_classifications=secondary_classifications,
            confidence_score=confidence,
            reasoning=self._generate_classification_reasoning(votes, confidence_factors),
            confidence_factors=confidence_factors,
            alternative_classifications=alternative_classifications
        )

    def _llm_dominant_classification(self, llm_entry, static_analysis,
                                   confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """LLM-dominant classification with static analysis as validation"""

        primary_classification = llm_entry.primary_classification
        confidence = llm_entry.confidence_score

        # Boost confidence if static analysis agrees
        if confidence_factors.agreement_confidence > 0.7:
            confidence = min(1.0, confidence + 0.1)

        return ClassificationResult(
            primary_classification=primary_classification,
            secondary_classifications=llm_entry.secondary_classifications,
            confidence_score=confidence,
            reasoning=["LLM-dominant classification with static validation"],
            confidence_factors=confidence_factors,
            alternative_classifications=[]
        )

    def _consensus_classification(self, llm_entry, static_analysis,
                                confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """Consensus-based classification requiring agreement"""

        # Use weighted voting but require minimum consensus
        weighted_result = self._weighted_voting_classification(llm_entry, static_analysis, confidence_factors)

        if confidence_factors.agreement_confidence < self.config['consensus_threshold']:
            # Low agreement - use LLM but reduce confidence
            return ClassificationResult(
                primary_classification=llm_entry.primary_classification,
                secondary_classifications=llm_entry.secondary_classifications,
                confidence_score=llm_entry.confidence_score * 0.7,
                reasoning=["Low consensus - using LLM with reduced confidence"],
                confidence_factors=confidence_factors,
                alternative_classifications=weighted_result.alternative_classifications
            )

        return weighted_result

    def _static_dominant_classification(self, llm_entry, static_analysis,
                                      confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """Static analysis dominant classification"""
        # This would implement static-dominant logic
        # For now, fall back to weighted voting
        return self._weighted_voting_classification(llm_entry, static_analysis, confidence_factors)

    def _adaptive_confidence_classification(self, llm_entry, static_analysis,
                                          confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """Adaptive confidence-based classification"""
        # This would implement adaptive confidence logic
        # For now, fall back to weighted voting
        return self._weighted_voting_classification(llm_entry, static_analysis, confidence_factors)

    def _get_pattern_based_classification(self, pattern_data: Dict[str, Any]) -> Dict[str, float]:
        """Get classification suggestions from pattern analysis"""
        votes = defaultdict(float)

        patterns = pattern_data.get('patterns', [])
        for pattern in patterns:
            pattern_name = pattern.get('pattern_name', '').lower()

            # Map patterns to categories
            if 'singleton' in pattern_name:
                votes['utility'] += 0.8
            elif 'factory' in pattern_name:
                votes['utility'] += 0.7
            elif 'observer' in pattern_name:
                votes['orchestration'] += 0.6
            elif 'strategy' in pattern_name:
                votes['utility'] += 0.6
            elif 'decorator' in pattern_name:
                votes['utility'] += 0.7

        return dict(votes)

    def _get_quality_based_classification(self, quality_data: Dict[str, Any]) -> str:
        """Get classification suggestion from quality analysis"""
        score = quality_data.get('overall_score', 0.5)

        if score < 0.6:
            return 'needs_refactoring'
        else:
            return 'utility'  # Default assumption

    def _generate_classification_reasoning(self, votes: Dict[str, float],
                                         confidence_factors: ConfidenceFactors) -> List[str]:
        """Generate reasoning for classification decision"""
        reasoning = []

        # Add vote information
        if votes:
            winner = max(votes.items(), key=lambda x: x[1])
            reasoning.append(f"Primary: {winner[0]} (score: {winner[1]:.3f})")

        # Add confidence factor information
        factors = []
        if confidence_factors.llm_confidence > 0.5:
            factors.append("LLM")
        if confidence_factors.semantic_confidence > 0.5:
            factors.append("semantic")
        if confidence_factors.pattern_confidence > 0.5:
            factors.append("pattern")
        if confidence_factors.quality_confidence > 0.5:
            factors.append("quality")

        if factors:
            reasoning.append(f"High confidence from: {', '.join(factors)}")

        return reasoning
