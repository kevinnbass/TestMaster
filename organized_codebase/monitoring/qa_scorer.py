"""
Quality Scoring System Component
=================================

Calculates quality scores for agent performance.
Part of modularized agent_qa system.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
import statistics

from .qa_base import (
    ScoreCategory, ScoreWeight, ScoreBreakdown,
    QualityScore, QualityLevel, QualityMetric
)


class QualityScorer:
    """Calculates and tracks quality scores."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quality scorer."""
        self.config = config or {}
        
        # Score weights
        self._weights = self._initialize_weights()
        
        # Score history
        self._score_history: List[QualityScore] = []
        self._category_scores: Dict[ScoreCategory, List[float]] = defaultdict(list)
        
        # Scoring parameters
        self._score_threshold_excellent = 90
        self._score_threshold_good = 75
        self._score_threshold_satisfactory = 60
        self._score_threshold_poor = 40
    
    def _initialize_weights(self) -> Dict[ScoreCategory, ScoreWeight]:
        """Initialize default score weights."""
        default_weights = {
            ScoreCategory.FUNCTIONALITY: ScoreWeight(ScoreCategory.FUNCTIONALITY, 0.25),
            ScoreCategory.RELIABILITY: ScoreWeight(ScoreCategory.RELIABILITY, 0.20),
            ScoreCategory.PERFORMANCE: ScoreWeight(ScoreCategory.PERFORMANCE, 0.20),
            ScoreCategory.SECURITY: ScoreWeight(ScoreCategory.SECURITY, 0.15),
            ScoreCategory.MAINTAINABILITY: ScoreWeight(ScoreCategory.MAINTAINABILITY, 0.10),
            ScoreCategory.USABILITY: ScoreWeight(ScoreCategory.USABILITY, 0.10)
        }
        
        # Override with config if provided
        if 'score_weights' in self.config:
            for category, weight in self.config['score_weights'].items():
                if category in default_weights:
                    default_weights[category].weight = weight
        
        return default_weights
    
    def calculate_score(self, metrics: List[QualityMetric]) -> QualityScore:
        """Calculate quality score from metrics."""
        # Group metrics by category
        category_metrics = defaultdict(list)
        for metric in metrics:
            category_metrics[metric.category].append(metric.value)
        
        # Calculate score for each category
        breakdowns = []
        total_weighted_score = 0
        total_weight = 0
        
        for category, weight_config in self._weights.items():
            if not weight_config.enabled:
                continue
            
            category_values = category_metrics.get(category, [])
            if category_values:
                # Calculate category score (normalize to 0-100)
                category_score = self._calculate_category_score(category, category_values)
                weighted_score = category_score * weight_config.weight
                
                breakdown = ScoreBreakdown(
                    category=category,
                    score=category_score,
                    weight=weight_config.weight,
                    weighted_score=weighted_score,
                    details={
                        'sample_count': len(category_values),
                        'mean': statistics.mean(category_values),
                        'median': statistics.median(category_values)
                    }
                )
                breakdowns.append(breakdown)
                
                total_weighted_score += weighted_score
                total_weight += weight_config.weight
        
        # Calculate total score
        total_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine quality level
        quality_level = self._determine_quality_level(total_score)
        
        # Create quality score
        quality_score = QualityScore(
            total_score=total_score,
            quality_level=quality_level,
            breakdown=breakdowns,
            timestamp=datetime.now(),
            metadata={'metrics_count': len(metrics)}
        )
        
        # Store in history
        self._score_history.append(quality_score)
        for breakdown in breakdowns:
            self._category_scores[breakdown.category].append(breakdown.score)
        
        return quality_score
    
    def _calculate_category_score(self, category: ScoreCategory, values: List[float]) -> float:
        """Calculate score for a specific category."""
        if not values:
            return 0.0
        
        # Different scoring logic per category
        if category == ScoreCategory.FUNCTIONALITY:
            # Higher is better (e.g., features completed)
            return min(100, statistics.mean(values))
        
        elif category == ScoreCategory.RELIABILITY:
            # Success rate percentage
            return min(100, statistics.mean(values))
        
        elif category == ScoreCategory.PERFORMANCE:
            # Inverse scoring for response time (lower is better)
            # Assume values are in milliseconds, target is 100ms
            target = 100
            mean_time = statistics.mean(values)
            if mean_time <= target:
                return 100
            else:
                return max(0, 100 - ((mean_time - target) / target * 20))
        
        elif category == ScoreCategory.SECURITY:
            # Percentage of security checks passed
            return min(100, statistics.mean(values))
        
        elif category == ScoreCategory.MAINTAINABILITY:
            # Code quality metrics (higher is better)
            return min(100, statistics.mean(values))
        
        elif category == ScoreCategory.USABILITY:
            # User satisfaction or ease of use metrics
            return min(100, statistics.mean(values))
        
        return statistics.mean(values)
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= self._score_threshold_excellent:
            return QualityLevel.EXCELLENT
        elif score >= self._score_threshold_good:
            return QualityLevel.GOOD
        elif score >= self._score_threshold_satisfactory:
            return QualityLevel.SATISFACTORY
        elif score >= self._score_threshold_poor:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def update_weight(self, category: ScoreCategory, weight: float):
        """Update weight for a category."""
        if category in self._weights:
            self._weights[category].weight = weight
            # Normalize weights
            self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        total = sum(w.weight for w in self._weights.values() if w.enabled)
        if total > 0:
            for weight in self._weights.values():
                if weight.enabled:
                    weight.weight = weight.weight / total
    
    def get_score_trends(self, window: int = 10) -> Dict[str, Any]:
        """Get score trends over recent history."""
        recent_scores = self._score_history[-window:] if len(self._score_history) > window else self._score_history
        
        if not recent_scores:
            return {}
        
        trends = {
            'total_scores': [s.total_score for s in recent_scores],
            'quality_levels': [s.quality_level.value for s in recent_scores],
            'trend_direction': self._calculate_trend_direction(recent_scores),
            'average_score': statistics.mean([s.total_score for s in recent_scores])
        }
        
        # Category trends
        for category in ScoreCategory:
            if category in self._category_scores:
                recent_category = self._category_scores[category][-window:]
                if recent_category:
                    trends[f'{category.value}_trend'] = {
                        'values': recent_category,
                        'mean': statistics.mean(recent_category),
                        'trend': self._calculate_simple_trend(recent_category)
                    }
        
        return trends
    
    def _calculate_trend_direction(self, scores: List[QualityScore]) -> str:
        """Calculate overall trend direction."""
        if len(scores) < 2:
            return "stable"
        
        values = [s.total_score for s in scores]
        trend = self._calculate_simple_trend(values)
        
        if trend > 0.1:
            return "improving"
        elif trend < -0.1:
            return "degrading"
        else:
            return "stable"
    
    def _calculate_simple_trend(self, values: List[float]) -> float:
        """Calculate simple trend coefficient."""
        if len(values) < 2:
            return 0.0
        
        # Simple difference between last half and first half averages
        mid = len(values) // 2
        first_half = statistics.mean(values[:mid])
        second_half = statistics.mean(values[mid:])
        
        if first_half == 0:
            return 0.0
        
        return (second_half - first_half) / first_half
    
    def get_latest_score(self) -> Optional[QualityScore]:
        """Get the most recent quality score."""
        return self._score_history[-1] if self._score_history else None
    
    def get_average_score(self, window: Optional[int] = None) -> float:
        """Get average score over history window."""
        scores = self._score_history[-window:] if window else self._score_history
        if not scores:
            return 0.0
        return statistics.mean([s.total_score for s in scores])


# Export
__all__ = ['QualityScorer']