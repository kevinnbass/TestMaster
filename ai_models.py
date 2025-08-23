#!/usr/bin/env python3
"""
AI Models Module
Extracted from ai_intelligence_engine.py via STEELCLAD Protocol

Data structures and model definitions for AI Intelligence Engine.
"""

from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class AIModel:
    """AI model configuration and state"""
    model_id: str
    model_type: str  # 'classification', 'regression', 'clustering', 'anomaly'
    version: float
    accuracy: float
    training_samples: int
    last_trained: datetime
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]

@dataclass
class IntelligentInsight:
    """AI-generated insight"""
    insight_id: str
    category: str  # 'performance', 'security', 'optimization', 'prediction'
    title: str
    description: str
    confidence: float  # 0-1
    impact_score: float  # 0-1
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    generated_at: datetime
    expires_at: Optional[datetime] = None

@dataclass
class PatternMatch:
    """Pattern matching result"""
    pattern_id: str
    pattern_name: str
    match_confidence: float
    matched_features: List[str]
    anomaly_score: float
    action_required: bool
    suggested_actions: List[str]