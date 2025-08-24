"""
Documentation Intelligence Metrics

Core metrics and data structures for documentation analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DocumentationType(Enum):
    """Types of documentation for intelligence analysis."""
    API_DOCUMENTATION = "api_documentation"
    CODE_DOCUMENTATION = "code_documentation"
    ARCHITECTURE_DOCS = "architecture_docs"
    USER_GUIDES = "user_guides"
    TECHNICAL_SPECS = "technical_specs"
    COMPLIANCE_DOCS = "compliance_docs"
    SECURITY_DOCS = "security_docs"


class IntelligenceMetric(Enum):
    """Documentation intelligence metrics."""
    READABILITY_SCORE = "readability_score"
    COMPLETENESS_INDEX = "completeness_index"
    ACCURACY_RATING = "accuracy_rating"
    CONSISTENCY_SCORE = "consistency_score"
    USEFULNESS_INDEX = "usefulness_index"
    MAINTENANCE_BURDEN = "maintenance_burden"
    STAKEHOLDER_SATISFACTION = "stakeholder_satisfaction"


class OptimizationPriority(Enum):
    """Priority levels for optimization recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


@dataclass
class DocumentationMetrics:
    """Comprehensive metrics for documentation analysis."""
    document_id: str
    document_type: DocumentationType
    analysis_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Quality metrics
    readability_score: float = 0.0
    completeness_index: float = 0.0
    accuracy_rating: float = 0.0
    consistency_score: float = 0.0
    usefulness_index: float = 0.0
    maintenance_burden: float = 0.0
    
    # Content analysis
    word_count: int = 0
    section_count: int = 0
    code_example_count: int = 0
    diagram_count: int = 0
    external_link_count: int = 0
    
    # Technical metrics
    api_coverage: float = 0.0
    code_coverage: float = 0.0
    example_accuracy: float = 0.0
    
    # Metadata
    last_updated: Optional[str] = None
    author_count: int = 0
    review_count: int = 0
    
    def calculate_overall_score(self) -> float:
        """Calculate overall documentation quality score."""
        try:
            core_metrics = [
                self.readability_score,
                self.completeness_index,
                self.accuracy_rating,
                self.consistency_score,
                self.usefulness_index
            ]
            
            # Weight core metrics equally
            base_score = sum(core_metrics) / len(core_metrics)
            
            # Apply maintenance burden penalty
            penalty = min(self.maintenance_burden * 0.1, 0.2)  # Max 20% penalty
            
            return max(0.0, min(100.0, base_score - penalty))
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0
    
    def get_metric_by_type(self, metric_type: IntelligenceMetric) -> float:
        """Get metric value by type."""
        try:
            metric_map = {
                IntelligenceMetric.READABILITY_SCORE: self.readability_score,
                IntelligenceMetric.COMPLETENESS_INDEX: self.completeness_index,
                IntelligenceMetric.ACCURACY_RATING: self.accuracy_rating,
                IntelligenceMetric.CONSISTENCY_SCORE: self.consistency_score,
                IntelligenceMetric.USEFULNESS_INDEX: self.usefulness_index,
                IntelligenceMetric.MAINTENANCE_BURDEN: self.maintenance_burden,
                IntelligenceMetric.STAKEHOLDER_SATISFACTION: self.calculate_overall_score()
            }
            return metric_map.get(metric_type, 0.0)
        except Exception as e:
            logger.error(f"Error getting metric {metric_type}: {e}")
            return 0.0


@dataclass
class OptimizationRecommendation:
    """Represents a documentation optimization recommendation."""
    recommendation_id: str
    priority: OptimizationPriority
    category: str
    title: str
    description: str
    impact_score: float
    effort_estimate: int  # Hours
    
    # Specific guidance
    current_state: str
    target_state: str
    action_items: List[str] = field(default_factory=list)
    
    # Context
    affected_sections: List[str] = field(default_factory=list)
    stakeholder_groups: List[str] = field(default_factory=list)
    
    # Metrics
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    
    def calculate_roi(self) -> float:
        """Calculate return on investment for this recommendation."""
        try:
            if self.effort_estimate == 0:
                return float('inf') if self.impact_score > 0 else 0.0
            
            return self.impact_score / self.effort_estimate
        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return 0.0


@dataclass
class IntelligenceInsight:
    """Represents an AI-generated insight about documentation."""
    insight_id: str
    insight_type: str
    confidence: float
    title: str
    description: str
    
    # Supporting data
    evidence: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    
    # Actionability
    actionable: bool = False
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_version: Optional[str] = None


@dataclass
class TrendAnalysis:
    """Analysis of documentation trends over time."""
    trend_id: str
    metric_type: IntelligenceMetric
    time_period: str
    
    # Trend data
    trend_direction: str  # "improving", "declining", "stable"
    change_rate: float    # Change per time unit
    confidence: float     # Confidence in trend detection
    
    # Data points
    historical_values: List[Tuple[str, float]] = field(default_factory=list)  # (timestamp, value)
    
    # Analysis
    significant_changes: List[Dict[str, Any]] = field(default_factory=list)
    seasonality_detected: bool = False
    
    def get_current_value(self) -> Optional[float]:
        """Get the most recent value."""
        try:
            if self.historical_values:
                return self.historical_values[-1][1]
            return None
        except Exception as e:
            logger.error(f"Error getting current value: {e}")
            return None
    
    def get_value_change(self, periods: int = 1) -> Optional[float]:
        """Get change over specified periods."""
        try:
            if len(self.historical_values) <= periods:
                return None
            
            current = self.historical_values[-1][1]
            previous = self.historical_values[-(periods + 1)][1]
            
            return current - previous
        except Exception as e:
            logger.error(f"Error calculating value change: {e}")
            return None


class MetricsCalculator:
    """Utility class for calculating documentation metrics."""
    
    @staticmethod
    def calculate_readability_score(content: str) -> float:
        """Calculate readability score using multiple algorithms."""
        try:
            if not content:
                return 0.0
            
            # Simple readability heuristics
            sentences = len([s for s in content.split('.') if s.strip()])
            words = len(content.split())
            
            if sentences == 0:
                return 0.0
            
            avg_sentence_length = words / sentences
            
            # Score based on average sentence length (optimal: 15-20 words)
            if 15 <= avg_sentence_length <= 20:
                base_score = 90.0
            elif 10 <= avg_sentence_length <= 25:
                base_score = 80.0
            elif 5 <= avg_sentence_length <= 30:
                base_score = 70.0
            else:
                base_score = 60.0
            
            # Adjust for content complexity
            complexity_indicators = [
                '(', ')', '[', ']', '{', '}',  # Code indicators
                'however', 'therefore', 'consequently',  # Complex transitions
                'implementation', 'configuration', 'optimization'  # Technical terms
            ]
            
            complexity_count = sum(content.lower().count(indicator) for indicator in complexity_indicators)
            complexity_penalty = min(complexity_count * 2, 20)  # Max 20 point penalty
            
            return max(0.0, min(100.0, base_score - complexity_penalty))
            
        except Exception as e:
            logger.error(f"Error calculating readability score: {e}")
            return 0.0
    
    @staticmethod
    def calculate_completeness_index(content: str, required_sections: List[str]) -> float:
        """Calculate completeness based on required sections."""
        try:
            if not required_sections:
                return 100.0
            
            content_lower = content.lower()
            sections_found = 0
            
            for section in required_sections:
                if section.lower() in content_lower:
                    sections_found += 1
            
            return (sections_found / len(required_sections)) * 100.0
            
        except Exception as e:
            logger.error(f"Error calculating completeness index: {e}")
            return 0.0
    
    @staticmethod
    def calculate_consistency_score(content: str) -> float:
        """Calculate consistency score based on formatting and terminology."""
        try:
            if not content:
                return 0.0
            
            # Check for consistent heading formats
            heading_patterns = [
                r'^#+\s+',  # Markdown headers
                r'^\d+\.\s+',  # Numbered sections
                r'^[A-Z][A-Za-z\s]+:$'  # Title case headers
            ]
            
            lines = content.split('\n')
            heading_lines = []
            
            for line in lines:
                line = line.strip()
                for pattern in heading_patterns:
                    if re.match(pattern, line):
                        heading_lines.append(line)
                        break
            
            if len(heading_lines) < 2:
                return 85.0  # Default score for short content
            
            # Simple consistency check - more sophisticated analysis would be better
            # This is a placeholder for demonstration
            base_score = 80.0
            
            # Check for consistent terminology (basic check)
            words = content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Only check longer words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Bonus for repeated terminology (indicates consistency)
            repeated_terms = len([w for w, count in word_freq.items() if count > 1])
            consistency_bonus = min(repeated_terms * 0.5, 15)  # Max 15 point bonus
            
            return min(100.0, base_score + consistency_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0.0