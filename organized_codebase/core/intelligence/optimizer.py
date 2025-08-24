"""
Documentation Intelligence Optimizer

Generates optimization recommendations and improvement strategies.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from .metrics import (
    DocumentationType, IntelligenceMetric, OptimizationPriority,
    DocumentationMetrics, OptimizationRecommendation, IntelligenceInsight,
    TrendAnalysis
)

logger = logging.getLogger(__name__)


class DocumentationOptimizer:
    """
    Generates intelligent optimization recommendations for documentation.
    Provides actionable insights and improvement strategies.
    """
    
    def __init__(self):
        """Initialize the documentation optimizer."""
        try:
            self.optimization_rules = self._load_optimization_rules()
            logger.info("Documentation Optimizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise
    
    async def generate_recommendations(self, 
                                     metrics: DocumentationMetrics,
                                     context: Optional[Dict[str, Any]] = None) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on metrics.
        
        Args:
            metrics: Document metrics
            context: Additional context for recommendations
            
        Returns:
            List of prioritized optimization recommendations
        """
        try:
            recommendations = []
            context = context or {}
            
            # Generate recommendations for each metric
            recommendations.extend(await self._analyze_readability(metrics, context))
            recommendations.extend(await self._analyze_completeness(metrics, context))
            recommendations.extend(await self._analyze_accuracy(metrics, context))
            recommendations.extend(await self._analyze_consistency(metrics, context))
            recommendations.extend(await self._analyze_usefulness(metrics, context))
            recommendations.extend(await self._analyze_maintenance_burden(metrics, context))
            
            # Sort by priority and impact
            recommendations.sort(key=lambda r: (
                self._priority_weight(r.priority),
                -r.impact_score,
                r.effort_estimate
            ))
            
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    async def generate_insights(self, 
                              metrics: DocumentationMetrics,
                              historical_data: Optional[List[DocumentationMetrics]] = None) -> List[IntelligenceInsight]:
        """
        Generate AI-powered insights about documentation.
        
        Args:
            metrics: Current document metrics
            historical_data: Historical metrics for trend analysis
            
        Returns:
            List of insights
        """
        try:
            insights = []
            
            # Quality insights
            insights.extend(await self._generate_quality_insights(metrics))
            
            # Content insights
            insights.extend(await self._generate_content_insights(metrics))
            
            # Trend insights (if historical data available)
            if historical_data:
                insights.extend(await self._generate_trend_insights(metrics, historical_data))
            
            # Performance insights
            insights.extend(await self._generate_performance_insights(metrics))
            
            # Filter insights by confidence threshold
            high_confidence_insights = [i for i in insights if i.confidence >= 0.7]
            
            logger.info(f"Generated {len(high_confidence_insights)} high-confidence insights")
            return high_confidence_insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []
    
    async def _analyze_readability(self, 
                                 metrics: DocumentationMetrics,
                                 context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze readability and generate recommendations."""
        try:
            recommendations = []
            
            if metrics.readability_score < 70:
                rec = OptimizationRecommendation(
                    recommendation_id=f"readability_{metrics.document_id}",
                    priority=OptimizationPriority.HIGH if metrics.readability_score < 50 else OptimizationPriority.MEDIUM,
                    category="Readability",
                    title="Improve Document Readability",
                    description="Document readability score is below recommended threshold",
                    impact_score=85 - metrics.readability_score,
                    effort_estimate=4,
                    current_state=f"Readability score: {metrics.readability_score:.1f}",
                    target_state="Readability score: 80+",
                    action_items=[
                        "Simplify complex sentences",
                        "Use shorter paragraphs",
                        "Add bullet points and lists",
                        "Improve section structure",
                        "Use active voice"
                    ],
                    affected_sections=["Content structure", "Writing style"],
                    stakeholder_groups=["Readers", "Content users"],
                    expected_improvement={
                        "readability_score": 15.0,
                        "user_satisfaction": 20.0
                    }
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing readability: {e}")
            return []
    
    async def _analyze_completeness(self, 
                                  metrics: DocumentationMetrics,
                                  context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze completeness and generate recommendations."""
        try:
            recommendations = []
            
            if metrics.completeness_index < 80:
                rec = OptimizationRecommendation(
                    recommendation_id=f"completeness_{metrics.document_id}",
                    priority=OptimizationPriority.HIGH if metrics.completeness_index < 60 else OptimizationPriority.MEDIUM,
                    category="Completeness",
                    title="Complete Missing Documentation Sections",
                    description="Documentation is missing required sections or content",
                    impact_score=90 - metrics.completeness_index,
                    effort_estimate=6,
                    current_state=f"Completeness: {metrics.completeness_index:.1f}%",
                    target_state="Completeness: 90%+",
                    action_items=[
                        "Add missing required sections",
                        "Expand incomplete sections",
                        "Add comprehensive examples",
                        "Include troubleshooting information",
                        "Add API reference details"
                    ],
                    affected_sections=["All sections"],
                    stakeholder_groups=["Developers", "Users", "Support team"],
                    expected_improvement={
                        "completeness_index": 20.0,
                        "usefulness_index": 15.0
                    }
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing completeness: {e}")
            return []
    
    async def _analyze_accuracy(self, 
                              metrics: DocumentationMetrics,
                              context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze accuracy and generate recommendations."""
        try:
            recommendations = []
            
            if metrics.accuracy_rating < 85:
                rec = OptimizationRecommendation(
                    recommendation_id=f"accuracy_{metrics.document_id}",
                    priority=OptimizationPriority.HIGH,
                    category="Accuracy",
                    title="Update and Verify Documentation Accuracy",
                    description="Documentation may contain outdated or inaccurate information",
                    impact_score=95 - metrics.accuracy_rating,
                    effort_estimate=8,
                    current_state=f"Accuracy rating: {metrics.accuracy_rating:.1f}",
                    target_state="Accuracy rating: 95+",
                    action_items=[
                        "Review and update outdated information",
                        "Verify code examples work correctly",
                        "Update version-specific information",
                        "Test all procedures and instructions",
                        "Remove deprecated content"
                    ],
                    affected_sections=["Technical content", "Examples", "Procedures"],
                    stakeholder_groups=["Developers", "Technical users"],
                    expected_improvement={
                        "accuracy_rating": 15.0,
                        "user_trust": 25.0
                    }
                )
                recommendations.append(rec)
            
            # Low example accuracy
            if metrics.example_accuracy < 90:
                rec = OptimizationRecommendation(
                    recommendation_id=f"examples_{metrics.document_id}",
                    priority=OptimizationPriority.HIGH,
                    category="Code Examples",
                    title="Fix Code Example Issues",
                    description="Code examples may have syntax errors or be outdated",
                    impact_score=50.0,
                    effort_estimate=3,
                    current_state=f"Example accuracy: {metrics.example_accuracy:.1f}%",
                    target_state="Example accuracy: 95%+",
                    action_items=[
                        "Test all code examples",
                        "Fix syntax errors",
                        "Update to current syntax",
                        "Add missing imports",
                        "Validate example outputs"
                    ],
                    affected_sections=["Code examples"],
                    stakeholder_groups=["Developers"],
                    expected_improvement={
                        "example_accuracy": 10.0,
                        "developer_productivity": 30.0
                    }
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing accuracy: {e}")
            return []
    
    async def _analyze_consistency(self, 
                                 metrics: DocumentationMetrics,
                                 context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze consistency and generate recommendations."""
        try:
            recommendations = []
            
            if metrics.consistency_score < 75:
                rec = OptimizationRecommendation(
                    recommendation_id=f"consistency_{metrics.document_id}",
                    priority=OptimizationPriority.MEDIUM,
                    category="Consistency",
                    title="Improve Documentation Consistency",
                    description="Documentation has inconsistent formatting, terminology, or style",
                    impact_score=40.0,
                    effort_estimate=5,
                    current_state=f"Consistency score: {metrics.consistency_score:.1f}",
                    target_state="Consistency score: 85+",
                    action_items=[
                        "Standardize heading formats",
                        "Use consistent terminology",
                        "Apply uniform code formatting",
                        "Standardize section structure",
                        "Create style guide"
                    ],
                    affected_sections=["Formatting", "Terminology"],
                    stakeholder_groups=["Writers", "Editors"],
                    expected_improvement={
                        "consistency_score": 15.0,
                        "professional_appearance": 20.0
                    }
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing consistency: {e}")
            return []
    
    async def _analyze_usefulness(self, 
                                metrics: DocumentationMetrics,
                                context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze usefulness and generate recommendations."""
        try:
            recommendations = []
            
            if metrics.usefulness_index < 75:
                rec = OptimizationRecommendation(
                    recommendation_id=f"usefulness_{metrics.document_id}",
                    priority=OptimizationPriority.MEDIUM,
                    category="Usefulness",
                    title="Enhance Documentation Usefulness",
                    description="Documentation could be more practical and useful for users",
                    impact_score=80 - metrics.usefulness_index,
                    effort_estimate=6,
                    current_state=f"Usefulness index: {metrics.usefulness_index:.1f}",
                    target_state="Usefulness index: 85+",
                    action_items=[
                        "Add more practical examples",
                        "Include step-by-step tutorials",
                        "Add troubleshooting section",
                        "Include best practices",
                        "Add common use cases"
                    ],
                    affected_sections=["Examples", "Tutorials", "Guidance"],
                    stakeholder_groups=["End users", "Developers"],
                    expected_improvement={
                        "usefulness_index": 15.0,
                        "user_productivity": 25.0
                    }
                )
                recommendations.append(rec)
            
            # Low code example count
            if metrics.code_example_count < 3 and metrics.document_type in [
                DocumentationType.API_DOCUMENTATION, 
                DocumentationType.CODE_DOCUMENTATION
            ]:
                rec = OptimizationRecommendation(
                    recommendation_id=f"more_examples_{metrics.document_id}",
                    priority=OptimizationPriority.MEDIUM,
                    category="Examples",
                    title="Add More Code Examples",
                    description="Documentation lacks sufficient code examples",
                    impact_score=35.0,
                    effort_estimate=4,
                    current_state=f"Code examples: {metrics.code_example_count}",
                    target_state="Code examples: 5+",
                    action_items=[
                        "Add basic usage examples",
                        "Include advanced examples",
                        "Add error handling examples",
                        "Show integration patterns",
                        "Include complete workflows"
                    ],
                    affected_sections=["Examples", "Usage"],
                    stakeholder_groups=["Developers"],
                    expected_improvement={
                        "usefulness_index": 20.0,
                        "developer_adoption": 30.0
                    }
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing usefulness: {e}")
            return []
    
    async def _analyze_maintenance_burden(self, 
                                        metrics: DocumentationMetrics,
                                        context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze maintenance burden and generate recommendations."""
        try:
            recommendations = []
            
            if metrics.maintenance_burden > 30:
                rec = OptimizationRecommendation(
                    recommendation_id=f"maintenance_{metrics.document_id}",
                    priority=OptimizationPriority.MEDIUM,
                    category="Maintenance",
                    title="Reduce Documentation Maintenance Burden",
                    description="Documentation has high maintenance requirements",
                    impact_score=metrics.maintenance_burden * 0.8,
                    effort_estimate=5,
                    current_state=f"Maintenance burden: {metrics.maintenance_burden:.1f}",
                    target_state="Maintenance burden: <20",
                    action_items=[
                        "Remove hardcoded values",
                        "Use generic examples",
                        "Automate version updates",
                        "Remove temporary content",
                        "Fix TODO items"
                    ],
                    affected_sections=["All content"],
                    stakeholder_groups=["Documentation team"],
                    expected_improvement={
                        "maintenance_burden": -15.0,
                        "team_efficiency": 20.0
                    }
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing maintenance burden: {e}")
            return []
    
    async def _generate_quality_insights(self, metrics: DocumentationMetrics) -> List[IntelligenceInsight]:
        """Generate quality-related insights."""
        try:
            insights = []
            
            overall_score = metrics.calculate_overall_score()
            
            if overall_score >= 90:
                insight = IntelligenceInsight(
                    insight_id=f"quality_excellent_{metrics.document_id}",
                    insight_type="quality_assessment",
                    confidence=0.9,
                    title="Excellent Documentation Quality",
                    description="This documentation demonstrates exceptional quality across all metrics",
                    evidence=[
                        f"Overall score: {overall_score:.1f}",
                        f"High readability: {metrics.readability_score:.1f}",
                        f"Complete content: {metrics.completeness_index:.1f}%"
                    ],
                    actionable=False
                )
                insights.append(insight)
            elif overall_score < 60:
                insight = IntelligenceInsight(
                    insight_id=f"quality_poor_{metrics.document_id}",
                    insight_type="quality_assessment",
                    confidence=0.95,
                    title="Documentation Needs Significant Improvement",
                    description="Multiple quality issues require immediate attention",
                    evidence=[
                        f"Low overall score: {overall_score:.1f}",
                        f"Below-par readability: {metrics.readability_score:.1f}",
                        f"Incomplete content: {metrics.completeness_index:.1f}%"
                    ],
                    actionable=True,
                    recommendations=[
                        "Prioritize completeness improvements",
                        "Focus on readability enhancements",
                        "Conduct comprehensive content review"
                    ]
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating quality insights: {e}")
            return []
    
    async def _generate_content_insights(self, metrics: DocumentationMetrics) -> List[IntelligenceInsight]:
        """Generate content-related insights."""
        try:
            insights = []
            
            # Content length analysis
            if metrics.word_count > 5000:
                insight = IntelligenceInsight(
                    insight_id=f"content_long_{metrics.document_id}",
                    insight_type="content_analysis",
                    confidence=0.8,
                    title="Document May Be Too Long",
                    description="Very long documents can be overwhelming for users",
                    evidence=[f"Word count: {metrics.word_count}"],
                    actionable=True,
                    recommendations=[
                        "Consider breaking into multiple documents",
                        "Add clear navigation structure",
                        "Use summary sections"
                    ]
                )
                insights.append(insight)
            
            # Code example density
            if metrics.code_example_count == 0 and metrics.document_type in [
                DocumentationType.API_DOCUMENTATION, 
                DocumentationType.CODE_DOCUMENTATION
            ]:
                insight = IntelligenceInsight(
                    insight_id=f"content_no_examples_{metrics.document_id}",
                    insight_type="content_analysis",
                    confidence=0.95,
                    title="Missing Code Examples",
                    description="Technical documentation should include code examples",
                    evidence=["No code examples found"],
                    actionable=True,
                    recommendations=[
                        "Add basic usage examples",
                        "Include error handling patterns",
                        "Show integration examples"
                    ]
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating content insights: {e}")
            return []
    
    async def _generate_trend_insights(self, 
                                     current_metrics: DocumentationMetrics,
                                     historical_data: List[DocumentationMetrics]) -> List[IntelligenceInsight]:
        """Generate trend-based insights."""
        try:
            insights = []
            
            if len(historical_data) < 2:
                return insights
            
            # Analyze quality trend
            historical_scores = [m.calculate_overall_score() for m in historical_data]
            current_score = current_metrics.calculate_overall_score()
            
            if len(historical_scores) >= 3:
                recent_trend = current_score - historical_scores[-2]
                
                if recent_trend > 10:
                    insight = IntelligenceInsight(
                        insight_id=f"trend_improving_{current_metrics.document_id}",
                        insight_type="trend_analysis",
                        confidence=0.85,
                        title="Documentation Quality Improving",
                        description="Significant improvement in documentation quality detected",
                        evidence=[
                            f"Score increased by {recent_trend:.1f} points",
                            f"Current score: {current_score:.1f}"
                        ],
                        actionable=False
                    )
                    insights.append(insight)
                elif recent_trend < -10:
                    insight = IntelligenceInsight(
                        insight_id=f"trend_declining_{current_metrics.document_id}",
                        insight_type="trend_analysis",
                        confidence=0.85,
                        title="Documentation Quality Declining",
                        description="Concerning decline in documentation quality",
                        evidence=[
                            f"Score decreased by {abs(recent_trend):.1f} points",
                            f"Current score: {current_score:.1f}"
                        ],
                        actionable=True,
                        recommendations=[
                            "Review recent changes",
                            "Implement quality checks",
                            "Schedule content audit"
                        ]
                    )
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating trend insights: {e}")
            return []
    
    async def _generate_performance_insights(self, metrics: DocumentationMetrics) -> List[IntelligenceInsight]:
        """Generate performance-related insights."""
        try:
            insights = []
            
            # API coverage insight
            if metrics.api_coverage < 50 and metrics.document_type == DocumentationType.API_DOCUMENTATION:
                insight = IntelligenceInsight(
                    insight_id=f"performance_api_coverage_{metrics.document_id}",
                    insight_type="performance_analysis",
                    confidence=0.8,
                    title="Low API Coverage",
                    description="API documentation doesn't cover all available endpoints",
                    evidence=[f"API coverage: {metrics.api_coverage:.1f}%"],
                    actionable=True,
                    recommendations=[
                        "Audit available API endpoints",
                        "Document missing endpoints",
                        "Automate API documentation generation"
                    ]
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
            return []
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules configuration."""
        return {
            "readability_threshold": 70,
            "completeness_threshold": 80,
            "accuracy_threshold": 85,
            "consistency_threshold": 75,
            "usefulness_threshold": 75,
            "maintenance_threshold": 30
        }
    
    def _priority_weight(self, priority: OptimizationPriority) -> int:
        """Convert priority to numeric weight for sorting."""
        weight_map = {
            OptimizationPriority.CRITICAL: 0,
            OptimizationPriority.HIGH: 1,
            OptimizationPriority.MEDIUM: 2,
            OptimizationPriority.LOW: 3,
            OptimizationPriority.OPTIONAL: 4
        }
        return weight_map.get(priority, 5)