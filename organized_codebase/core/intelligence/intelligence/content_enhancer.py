"""
Content Enhancer

AI-powered content enhancement for documentation optimization.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

from ..metrics import (
    DocumentationType, IntelligenceMetric, OptimizationPriority,
    DocumentationMetrics, OptimizationRecommendation, IntelligenceInsight
)

logger = logging.getLogger(__name__)


class ContentEnhancer:
    """
    AI-powered content enhancement system for improving documentation
    quality through intelligent content analysis and optimization.
    """
    
    def __init__(self):
        """Initialize the content enhancer."""
        try:
            self.enhancement_patterns = self._load_enhancement_patterns()
            self.content_rules = self._load_content_rules()
            logger.info("Content Enhancer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize content enhancer: {e}")
            raise
    
    async def analyze_usefulness(self, 
                                metrics: DocumentationMetrics, 
                                context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate usefulness optimization recommendations."""
        try:
            recommendations = []
            
            if metrics.usefulness_index < 70:
                recommendations.append(OptimizationRecommendation(
                    metric=IntelligenceMetric.USEFULNESS,
                    priority=OptimizationPriority.HIGH,
                    title="Increase Content Usefulness",
                    description="Document lacks practical value for users. Add more actionable content.",
                    action_items=[
                        "Add step-by-step tutorials",
                        "Include real-world examples",
                        "Add troubleshooting section",
                        "Include best practices and common patterns",
                        "Add FAQ section for common questions"
                    ],
                    impact_score=85,
                    effort_estimate=5,
                    success_metrics=["Usefulness index > 80", "Added 5+ practical examples"]
                ))
            elif metrics.usefulness_index < 80:
                recommendations.append(OptimizationRecommendation(
                    metric=IntelligenceMetric.USEFULNESS,
                    priority=OptimizationPriority.MEDIUM,
                    title="Enhance Practical Value",
                    description="Good usefulness but can be enhanced with more practical content.",
                    action_items=[
                        "Add more code examples",
                        "Include performance tips",
                        "Add links to related resources"
                    ],
                    impact_score=70,
                    effort_estimate=3,
                    success_metrics=["Usefulness index > 85"]
                ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing usefulness: {e}")
            return []
    
    async def enhance_content_structure(self, 
                                       content: str, 
                                       document_type: DocumentationType) -> Dict[str, Any]:
        """Analyze and enhance content structure."""
        try:
            structure_analysis = {
                'sections': self._analyze_sections(content),
                'headings': self._analyze_headings(content),
                'flow': self._analyze_content_flow(content),
                'recommendations': []
            }
            
            # Analyze section balance
            if structure_analysis['sections']['imbalance_score'] > 70:
                structure_analysis['recommendations'].append({
                    'type': 'structure',
                    'priority': 'high',
                    'description': 'Sections are imbalanced - some very long, others very short',
                    'suggestion': 'Break up long sections or expand short ones for better balance'
                })
            
            # Analyze heading hierarchy
            if structure_analysis['headings']['hierarchy_issues']:
                structure_analysis['recommendations'].append({
                    'type': 'headings',
                    'priority': 'medium', 
                    'description': 'Heading hierarchy has gaps or inconsistencies',
                    'suggestion': 'Fix heading levels to create proper hierarchy'
                })
            
            return structure_analysis
            
        except Exception as e:
            logger.error(f"Error enhancing content structure: {e}")
            return {'sections': {}, 'headings': {}, 'flow': {}, 'recommendations': []}
    
    async def generate_content_insights(self, 
                                       metrics: DocumentationMetrics) -> List[IntelligenceInsight]:
        """Generate AI-powered content insights."""
        try:
            insights = []
            
            # Content length insights
            if metrics.word_count < 500:
                insights.append(IntelligenceInsight(
                    insight_type="content_length",
                    title="Document May Be Too Brief",
                    description=f"With only {metrics.word_count} words, this document may not provide sufficient detail for users.",
                    confidence_score=0.8,
                    actionable_recommendations=[
                        "Consider expanding key sections",
                        "Add more examples and use cases",
                        "Include troubleshooting information"
                    ],
                    supporting_data={'word_count': metrics.word_count, 'recommended_minimum': 800}
                ))
            elif metrics.word_count > 5000:
                insights.append(IntelligenceInsight(
                    insight_type="content_length",
                    title="Document May Be Too Lengthy",
                    description=f"With {metrics.word_count} words, users might find this document overwhelming.",
                    confidence_score=0.7,
                    actionable_recommendations=[
                        "Consider breaking into multiple documents",
                        "Add table of contents for navigation",
                        "Create summary section"
                    ],
                    supporting_data={'word_count': metrics.word_count, 'recommended_maximum': 4000}
                ))
            
            # Code example insights
            if metrics.code_example_count == 0:
                insights.append(IntelligenceInsight(
                    insight_type="code_examples",
                    title="Missing Code Examples",
                    description="Technical documentation benefits significantly from code examples.",
                    confidence_score=0.9,
                    actionable_recommendations=[
                        "Add basic usage examples",
                        "Include common patterns",
                        "Show error handling examples"
                    ],
                    supporting_data={'current_examples': 0, 'recommended_minimum': 3}
                ))
            elif metrics.code_example_count < 3 and metrics.document_type == DocumentationType.API_DOCUMENTATION:
                insights.append(IntelligenceInsight(
                    insight_type="code_examples",
                    title="Limited Code Examples for API Documentation",
                    description="API documentation typically needs more code examples for different use cases.",
                    confidence_score=0.8,
                    actionable_recommendations=[
                        "Add examples for each major endpoint",
                        "Include authentication examples",
                        "Show error response examples"
                    ],
                    supporting_data={'current_examples': metrics.code_example_count, 'recommended_for_api': 5}
                ))
            
            # External link insights
            if metrics.external_link_count == 0:
                insights.append(IntelligenceInsight(
                    insight_type="external_links",
                    title="No External References",
                    description="Documentation often benefits from links to related resources.",
                    confidence_score=0.6,
                    actionable_recommendations=[
                        "Link to official documentation",
                        "Reference related tools or libraries",
                        "Include links to tutorials or guides"
                    ],
                    supporting_data={'current_links': 0}
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating content insights: {e}")
            return []
    
    def _load_enhancement_patterns(self) -> Dict[str, Any]:
        """Load content enhancement patterns."""
        try:
            return {
                'useful_phrases': [
                    'step-by-step', 'tutorial', 'example', 'how to',
                    'best practice', 'common pattern', 'troubleshooting'
                ],
                'structure_indicators': [
                    'overview', 'introduction', 'getting started', 'examples',
                    'advanced', 'reference', 'troubleshooting', 'faq'
                ],
                'quality_signals': [
                    'note:', 'important:', 'warning:', 'tip:', 'example:'
                ]
            }
        except Exception as e:
            logger.error(f"Error loading enhancement patterns: {e}")
            return {}
    
    def _load_content_rules(self) -> Dict[str, Any]:
        """Load content quality rules."""
        try:
            return {
                'min_section_words': 50,
                'max_section_words': 1000,
                'ideal_paragraph_sentences': (3, 6),
                'max_sentence_words': 25,
                'min_examples_per_concept': 1
            }
        except Exception as e:
            logger.error(f"Error loading content rules: {e}")
            return {}
    
    def _analyze_sections(self, content: str) -> Dict[str, Any]:
        """Analyze section structure and balance."""
        try:
            # Find sections by headers
            sections = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            
            if not sections:
                return {'count': 0, 'imbalance_score': 0, 'issues': []}
            
            # Split content by sections to analyze length
            section_parts = re.split(r'^#+\s+.+$', content, flags=re.MULTILINE)
            section_lengths = [len(part.split()) for part in section_parts[1:]]  # Skip content before first header
            
            if not section_lengths:
                return {'count': len(sections), 'imbalance_score': 0, 'issues': []}
            
            # Calculate imbalance score
            avg_length = sum(section_lengths) / len(section_lengths)
            variance = sum((length - avg_length) ** 2 for length in section_lengths) / len(section_lengths)
            imbalance_score = min(variance / avg_length if avg_length > 0 else 0, 100)
            
            issues = []
            for i, length in enumerate(section_lengths):
                if length < 20:
                    issues.append(f"Section {i+1} is very short ({length} words)")
                elif length > 800:
                    issues.append(f"Section {i+1} is very long ({length} words)")
            
            return {
                'count': len(sections),
                'lengths': section_lengths,
                'average_length': avg_length,
                'imbalance_score': imbalance_score,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sections: {e}")
            return {'count': 0, 'imbalance_score': 0, 'issues': []}
    
    def _analyze_headings(self, content: str) -> Dict[str, Any]:
        """Analyze heading structure and hierarchy."""
        try:
            # Find all headings with their levels
            headings = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
            
            if not headings:
                return {'count': 0, 'hierarchy_issues': [], 'max_depth': 0}
            
            levels = [len(heading[0]) for heading in headings]
            hierarchy_issues = []
            
            # Check for hierarchy gaps (e.g., # followed by ###)
            for i in range(1, len(levels)):
                if levels[i] - levels[i-1] > 1:
                    hierarchy_issues.append(f"Heading level gap at position {i+1}")
            
            return {
                'count': len(headings),
                'levels': levels,
                'max_depth': max(levels) if levels else 0,
                'hierarchy_issues': hierarchy_issues
            }
            
        except Exception as e:
            logger.error(f"Error analyzing headings: {e}")
            return {'count': 0, 'hierarchy_issues': [], 'max_depth': 0}
    
    def _analyze_content_flow(self, content: str) -> Dict[str, Any]:
        """Analyze content flow and logical progression."""
        try:
            paragraphs = content.split('\n\n')
            paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
            
            # Analyze paragraph consistency
            if paragraph_lengths:
                avg_paragraph_length = sum(paragraph_lengths) / len(paragraph_lengths)
                short_paragraphs = sum(1 for length in paragraph_lengths if length < 10)
                long_paragraphs = sum(1 for length in paragraph_lengths if length > 150)
            else:
                avg_paragraph_length = 0
                short_paragraphs = 0
                long_paragraphs = 0
            
            # Look for transition words and phrases
            transition_indicators = [
                'however', 'therefore', 'furthermore', 'additionally',
                'in contrast', 'similarly', 'for example', 'as a result'
            ]
            
            transition_count = sum(content.lower().count(indicator) for indicator in transition_indicators)
            
            return {
                'paragraph_count': len(paragraph_lengths),
                'average_paragraph_length': avg_paragraph_length,
                'short_paragraphs': short_paragraphs,
                'long_paragraphs': long_paragraphs,
                'transition_count': transition_count,
                'flow_score': min(100, transition_count * 10 + (100 - short_paragraphs * 5))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content flow: {e}")
            return {'paragraph_count': 0, 'flow_score': 0}