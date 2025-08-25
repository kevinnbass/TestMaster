#!/usr/bin/env python3
"""
Intelligence Integration Engine Data Models
===========================================

Data structures and models for the intelligence integration system.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class IntegrationMethod(Enum):
    """Methods for integrating multiple intelligence sources"""
    WEIGHTED_VOTING = "weighted_voting"
    CONSENSUS_WITH_FALLBACK = "consensus_with_fallback"
    LLM_DOMINANT = "llm_dominant"
    STATIC_DOMINANT = "static_dominant"
    ADAPTIVE_CONFIDENCE = "adaptive_confidence"


@dataclass
class ConfidenceFactors:
    """Confidence factors from different analysis sources"""
    llm_confidence: float = 0.0
    semantic_confidence: float = 0.0
    pattern_confidence: float = 0.0
    quality_confidence: float = 0.0
    relationship_confidence: float = 0.0
    agreement_confidence: float = 0.0

    def calculate_overall_confidence(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall confidence using weighted factors"""
        if weights is None:
            weights = {
                'llm_confidence': 0.35,
                'semantic_confidence': 0.20,
                'pattern_confidence': 0.15,
                'quality_confidence': 0.15,
                'relationship_confidence': 0.10,
                'agreement_confidence': 0.05
            }

        weighted_sum = sum(
            getattr(self, factor) * weight
            for factor, weight in weights.items()
            if hasattr(self, factor)
        )

        return min(1.0, max(0.0, weighted_sum))


@dataclass
class ClassificationResult:
    """Result of classification integration"""
    primary_classification: str
    secondary_classifications: List[str]
    confidence_score: float
    reasoning: List[str]
    confidence_factors: ConfidenceFactors
    alternative_classifications: List[Tuple[str, float]]


@dataclass
class ReorganizationRecommendation:
    """Specific reorganization recommendation"""
    source_path: str
    target_path: str
    rationale: str
    confidence: float
    priority: int
    risk_level: str
    dependencies: List[str]
    prerequisites: List[str]
    estimated_effort: int  # minutes
