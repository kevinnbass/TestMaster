"""
Scoring System for TestMaster Agent QA

Calculates comprehensive quality scores for agent operations.
"""

import threading
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..core.feature_flags import FeatureFlags
from .quality_inspector import QualityMetric

class ScoreCategory(Enum):
    """Score categories."""
    FUNCTIONALITY = "functionality"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    USABILITY = "usability"

@dataclass
class ScoreWeight:
    """Score weight configuration."""
    category: ScoreCategory
    weight: float
    description: str

@dataclass
class ScoreBreakdown:
    """Score breakdown by category."""
    category: ScoreCategory
    score: float
    weight: float
    weighted_score: float
    details: Dict[str, Any]

@dataclass
class QualityScore:
    """Quality score result."""
    agent_id: str
    overall_score: float
    breakdown: List[ScoreBreakdown]
    status: str
    grade: str = ""
    percentile: float = 0.0
    
    def __post_init__(self):
        self.grade = self._calculate_grade()
    
    def _calculate_grade(self) -> str:
        """Calculate letter grade from score."""
        if self.overall_score >= 0.95:
            return "A+"
        elif self.overall_score >= 0.9:
            return "A"
        elif self.overall_score >= 0.85:
            return "A-"
        elif self.overall_score >= 0.8:
            return "B+"
        elif self.overall_score >= 0.75:
            return "B"
        elif self.overall_score >= 0.7:
            return "B-"
        elif self.overall_score >= 0.65:
            return "C+"
        elif self.overall_score >= 0.6:
            return "C"
        elif self.overall_score >= 0.55:
            return "C-"
        elif self.overall_score >= 0.5:
            return "D"
        else:
            return "F"

class ScoringSystem:
    """Scoring system for agent quality assessment."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer1_test_foundation', 'agent_qa')
        self.lock = threading.RLock()
        self.scoring_history: Dict[str, List[QualityScore]] = {}
        self.score_weights = self._setup_default_weights()
        self.benchmarks: Dict[str, float] = {}
        
        if not self.enabled:
            return
        
        print("Scoring system initialized")
        print(f"   Score categories: {[w.category.value for w in self.score_weights]}")
    
    def _setup_default_weights(self) -> List[ScoreWeight]:
        """Setup default score weights."""
        return [
            ScoreWeight(
                category=ScoreCategory.FUNCTIONALITY,
                weight=0.25,
                description="Correctness and feature completeness"
            ),
            ScoreWeight(
                category=ScoreCategory.RELIABILITY,
                weight=0.20,
                description="Consistency and error handling"
            ),
            ScoreWeight(
                category=ScoreCategory.PERFORMANCE,
                weight=0.20,
                description="Speed and resource efficiency"
            ),
            ScoreWeight(
                category=ScoreCategory.SECURITY,
                weight=0.15,
                description="Security and vulnerability assessment"
            ),
            ScoreWeight(
                category=ScoreCategory.MAINTAINABILITY,
                weight=0.15,
                description="Code quality and maintainability"
            ),
            ScoreWeight(
                category=ScoreCategory.USABILITY,
                weight=0.05,
                description="User experience and accessibility"
            )
        ]
    
    def calculate_score(
        self,
        agent_id: str,
        quality_metrics: List[QualityMetric],
        custom_weights: Dict[str, float] = None
    ) -> QualityScore:
        """
        Calculate comprehensive quality score.
        
        Args:
            agent_id: Agent identifier
            quality_metrics: List of quality metrics
            custom_weights: Custom weights for categories
            
        Returns:
            Quality score with detailed breakdown
        """
        if not self.enabled:
            return QualityScore(agent_id, 0.0, [], "disabled")
        
        # Use custom weights if provided
        weights = self._apply_custom_weights(custom_weights) if custom_weights else self.score_weights
        
        # Calculate scores by category
        breakdown = []
        for weight in weights:
            category_score = self._calculate_category_score(quality_metrics, weight.category)
            weighted_score = category_score * weight.weight
            
            breakdown.append(ScoreBreakdown(
                category=weight.category,
                score=category_score,
                weight=weight.weight,
                weighted_score=weighted_score,
                details=self._get_category_details(quality_metrics, weight.category)
            ))
        
        # Calculate overall score
        overall_score = sum(b.weighted_score for b in breakdown)
        
        # Determine status
        status = self._determine_score_status(overall_score)
        
        # Calculate percentile
        percentile = self._calculate_percentile(agent_id, overall_score)
        
        score_result = QualityScore(
            agent_id=agent_id,
            overall_score=overall_score,
            breakdown=breakdown,
            status=status,
            percentile=percentile
        )
        
        # Store in history
        with self.lock:
            if agent_id not in self.scoring_history:
                self.scoring_history[agent_id] = []
            self.scoring_history[agent_id].append(score_result)
        
        print(f"Quality score calculated for {agent_id}: {overall_score:.3f} ({score_result.grade}) - {status}")
        
        return score_result
    
    def _calculate_category_score(self, metrics: List[QualityMetric], category: ScoreCategory) -> float:
        """Calculate score for a specific category."""
        category_metrics = self._filter_metrics_by_category(metrics, category)
        
        if not category_metrics:
            return 0.8  # Default score if no metrics available
        
        # Calculate weighted average of category metrics
        total_score = 0.0
        total_weight = 0.0
        
        for metric in category_metrics:
            weight = self._get_metric_weight(metric.name, category)
            total_score += metric.value * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _filter_metrics_by_category(self, metrics: List[QualityMetric], category: ScoreCategory) -> List[QualityMetric]:
        """Filter metrics by category."""
        category_mapping = {
            ScoreCategory.FUNCTIONALITY: ["syntax_validation", "semantic_analysis"],
            ScoreCategory.RELIABILITY: ["reliability_test", "error_handling"],
            ScoreCategory.PERFORMANCE: ["performance_test", "response_time"],
            ScoreCategory.SECURITY: ["security_scan", "vulnerability_check"],
            ScoreCategory.MAINTAINABILITY: ["code_quality", "documentation"],
            ScoreCategory.USABILITY: ["usability_test", "accessibility"]
        }
        
        relevant_names = category_mapping.get(category, [])
        return [m for m in metrics if m.name in relevant_names or category.value in m.name]
    
    def _get_metric_weight(self, metric_name: str, category: ScoreCategory) -> float:
        """Get weight for a specific metric within a category."""
        metric_weights = {
            "syntax_validation": 0.6,
            "semantic_analysis": 0.4,
            "performance_test": 0.7,
            "response_time": 0.3,
            "security_scan": 1.0,
            "reliability_test": 1.0
        }
        return metric_weights.get(metric_name, 1.0)
    
    def _get_category_details(self, metrics: List[QualityMetric], category: ScoreCategory) -> Dict[str, Any]:
        """Get detailed information for a category."""
        category_metrics = self._filter_metrics_by_category(metrics, category)
        
        return {
            "metrics_count": len(category_metrics),
            "metrics_passed": sum(1 for m in category_metrics if m.status == "pass"),
            "average_score": sum(m.value for m in category_metrics) / len(category_metrics) if category_metrics else 0.0,
            "metric_names": [m.name for m in category_metrics]
        }
    
    def _apply_custom_weights(self, custom_weights: Dict[str, float]) -> List[ScoreWeight]:
        """Apply custom weights to scoring categories."""
        updated_weights = []
        
        for weight in self.score_weights:
            new_weight_value = custom_weights.get(weight.category.value, weight.weight)
            updated_weights.append(ScoreWeight(
                category=weight.category,
                weight=new_weight_value,
                description=weight.description
            ))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(w.weight for w in updated_weights)
        if total_weight > 0:
            for weight in updated_weights:
                weight.weight = weight.weight / total_weight
        
        return updated_weights
    
    def _determine_score_status(self, overall_score: float) -> str:
        """Determine status from overall score."""
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "good"
        elif overall_score >= 0.7:
            return "satisfactory"
        elif overall_score >= 0.6:
            return "needs_improvement"
        else:
            return "poor"
    
    def _calculate_percentile(self, agent_id: str, current_score: float) -> float:
        """Calculate percentile ranking for the score."""
        # Get all historical scores
        all_scores = []
        for scores in self.scoring_history.values():
            all_scores.extend([s.overall_score for s in scores])
        
        if not all_scores:
            return 50.0  # Default percentile
        
        # Calculate percentile
        scores_below = sum(1 for score in all_scores if score < current_score)
        percentile = (scores_below / len(all_scores)) * 100
        
        return percentile
    
    def get_scoring_history(self, agent_id: str) -> List[QualityScore]:
        """Get scoring history for an agent."""
        with self.lock:
            return self.scoring_history.get(agent_id, [])
    
    def get_score_trends(self, agent_id: str) -> Dict[str, Any]:
        """Get score trends for an agent."""
        history = self.get_scoring_history(agent_id)
        if len(history) < 2:
            return {"status": "insufficient_data"}
        
        scores = [s.overall_score for s in history]
        recent_avg = sum(scores[-3:]) / min(3, len(scores))
        older_avg = sum(scores[:-3]) / max(1, len(scores) - 3) if len(scores) > 3 else scores[0]
        
        return {
            "trend": "improving" if recent_avg > older_avg else "declining",
            "latest_score": scores[-1],
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "score_variance": max(scores) - min(scores),
            "improvement_rate": (scores[-1] - scores[0]) / len(scores) if len(scores) > 1 else 0.0
        }
    
    def set_benchmark(self, benchmark_name: str, score: float):
        """Set a benchmark score for comparison."""
        self.benchmarks[benchmark_name] = score
        print(f"Benchmark set: {benchmark_name} = {score:.3f}")
    
    def compare_to_benchmark(self, agent_id: str, benchmark_name: str) -> Dict[str, Any]:
        """Compare agent score to benchmark."""
        history = self.get_scoring_history(agent_id)
        if not history:
            return {"status": "no_data"}
        
        latest_score = history[-1].overall_score
        benchmark_score = self.benchmarks.get(benchmark_name)
        
        if benchmark_score is None:
            return {"status": "benchmark_not_found"}
        
        difference = latest_score - benchmark_score
        return {
            "agent_score": latest_score,
            "benchmark_score": benchmark_score,
            "difference": difference,
            "performance": "above" if difference > 0 else "below" if difference < 0 else "equal"
        }

def get_scoring_system() -> ScoringSystem:
    """Get scoring system instance."""
    return ScoringSystem()