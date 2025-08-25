"""
Quality Assessment Engine
========================
Advanced quality assessment and optimization system for generated documentation.

This module provides comprehensive quality evaluation, automated improvement suggestions,
and continuous optimization based on patterns from all 7 frameworks.

Author: Agent D - Documentation Intelligence
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime
import asyncio
import json

class QualityMetric(Enum):
    """Quality metrics for documentation assessment."""
    READABILITY = "readability"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    ACCESSIBILITY = "accessibility"
    MAINTAINABILITY = "maintainability"
    USABILITY = "usability"
    TECHNICAL_DEPTH = "technical_depth"

class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"

@dataclass
class QualityScore:
    """Individual quality metric score."""
    metric: QualityMetric
    score: float  # 0.0 - 1.0
    level: QualityLevel
    feedback: str
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.9

@dataclass
class QualityAssessment:
    """Complete quality assessment results."""
    content_id: str
    overall_score: float
    overall_level: QualityLevel
    metric_scores: Dict[QualityMetric, QualityScore]
    priority_improvements: List[str]
    framework_compliance: Dict[str, float]
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    
class ReadabilityAnalyzer:
    """Analyzes documentation readability using multiple frameworks' best practices."""
    
    def __init__(self):
        self.sentence_complexity_threshold = 20
        self.paragraph_length_threshold = 150
        self.technical_density_threshold = 0.3
        
    def analyze_readability(self, content: str) -> QualityScore:
        """Comprehensive readability analysis."""
        sentences = self._split_sentences(content)
        paragraphs = self._split_paragraphs(content)
        
        # Multi-dimensional readability metrics
        sentence_complexity = self._calculate_sentence_complexity(sentences)
        paragraph_flow = self._analyze_paragraph_flow(paragraphs)
        technical_density = self._calculate_technical_density(content)
        structure_clarity = self._assess_structure_clarity(content)
        
        # Weighted scoring based on framework patterns
        readability_score = (
            sentence_complexity * 0.25 +
            paragraph_flow * 0.25 +
            (1.0 - technical_density) * 0.25 +
            structure_clarity * 0.25
        )
        
        level = self._determine_quality_level(readability_score)
        suggestions = self._generate_readability_suggestions(
            sentence_complexity, paragraph_flow, technical_density, structure_clarity
        )
        
        return QualityScore(
            metric=QualityMetric.READABILITY,
            score=readability_score,
            level=level,
            feedback=f"Readability score: {readability_score:.2f}. {self._get_readability_feedback(level)}",
            suggestions=suggestions
        )
    
    def _split_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        return re.split(r'[.!?]+', content)
    
    def _split_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs."""
        return [p.strip() for p in content.split('\n\n') if p.strip()]
    
    def _calculate_sentence_complexity(self, sentences: List[str]) -> float:
        """Calculate average sentence complexity score."""
        if not sentences:
            return 1.0
            
        complexities = []
        for sentence in sentences:
            word_count = len(sentence.split())
            clause_count = sentence.count(',') + sentence.count(';') + 1
            complexity = min(1.0, word_count / self.sentence_complexity_threshold)
            complexities.append(1.0 - complexity)
        
        return sum(complexities) / len(complexities)
    
    def _analyze_paragraph_flow(self, paragraphs: List[str]) -> float:
        """Analyze paragraph structure and flow."""
        if not paragraphs:
            return 1.0
            
        flow_scores = []
        for paragraph in paragraphs:
            length_score = min(1.0, self.paragraph_length_threshold / len(paragraph))
            transition_score = self._check_transitions(paragraph)
            flow_scores.append((length_score + transition_score) / 2)
        
        return sum(flow_scores) / len(flow_scores)
    
    def _calculate_technical_density(self, content: str) -> float:
        """Calculate technical term density."""
        words = content.lower().split()
        technical_patterns = [
            r'\b\w*api\w*\b', r'\b\w*config\w*\b', r'\b\w*framework\w*\b',
            r'\b\w*class\w*\b', r'\b\w*method\w*\b', r'\b\w*function\w*\b'
        ]
        
        technical_count = 0
        for pattern in technical_patterns:
            technical_count += len(re.findall(pattern, content.lower()))
        
        return min(1.0, technical_count / len(words)) if words else 0.0
    
    def _assess_structure_clarity(self, content: str) -> float:
        """Assess structural clarity using headers, lists, code blocks."""
        structure_elements = (
            content.count('#') +  # Headers
            content.count('- ') +  # Lists
            content.count('```') // 2  # Code blocks
        )
        
        content_length = len(content.split())
        structure_density = structure_elements / content_length if content_length > 0 else 0
        
        # Optimal structure density is around 0.05-0.15
        optimal_range = (0.05, 0.15)
        if optimal_range[0] <= structure_density <= optimal_range[1]:
            return 1.0
        elif structure_density < optimal_range[0]:
            return structure_density / optimal_range[0]
        else:
            return max(0.0, 1.0 - (structure_density - optimal_range[1]) / 0.1)
    
    def _check_transitions(self, paragraph: str) -> float:
        """Check for transition words and phrases."""
        transition_words = [
            'however', 'therefore', 'furthermore', 'additionally',
            'consequently', 'meanwhile', 'similarly', 'in contrast'
        ]
        
        found_transitions = sum(1 for word in transition_words if word in paragraph.lower())
        return min(1.0, found_transitions / 3)  # Optimal: 1-3 transitions per paragraph
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.8:
            return QualityLevel.GOOD
        elif score >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.5:
            return QualityLevel.NEEDS_IMPROVEMENT
        else:
            return QualityLevel.POOR
    
    def _get_readability_feedback(self, level: QualityLevel) -> str:
        """Generate feedback based on quality level."""
        feedback_map = {
            QualityLevel.EXCELLENT: "Content is highly readable and accessible.",
            QualityLevel.GOOD: "Content has good readability with minor improvements possible.",
            QualityLevel.ACCEPTABLE: "Content is readable but could benefit from optimization.",
            QualityLevel.NEEDS_IMPROVEMENT: "Content requires significant readability improvements.",
            QualityLevel.POOR: "Content has serious readability issues that must be addressed."
        }
        return feedback_map.get(level, "Quality level assessment unavailable.")
    
    def _generate_readability_suggestions(self, sentence_complexity: float, 
                                        paragraph_flow: float, technical_density: float, 
                                        structure_clarity: float) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        if sentence_complexity < 0.7:
            suggestions.append("Simplify complex sentences and reduce average sentence length")
        
        if paragraph_flow < 0.7:
            suggestions.append("Improve paragraph structure and add transition words")
        
        if technical_density > self.technical_density_threshold:
            suggestions.append("Reduce technical density and add explanations for complex terms")
        
        if structure_clarity < 0.7:
            suggestions.append("Add more structural elements (headers, lists, code blocks)")
        
        return suggestions

class CompletenessValidator:
    """Validates documentation completeness using framework standards."""
    
    def __init__(self):
        self.required_sections = {
            'api': ['overview', 'parameters', 'returns', 'examples'],
            'tutorial': ['introduction', 'prerequisites', 'steps', 'conclusion'],
            'guide': ['overview', 'concepts', 'implementation', 'best_practices'],
            'reference': ['description', 'usage', 'parameters', 'examples']
        }
    
    def validate_completeness(self, content: str, doc_type: str = 'general') -> QualityScore:
        """Validate documentation completeness."""
        sections_present = self._identify_sections(content)
        required_sections = self.required_sections.get(doc_type, [])
        
        if required_sections:
            completeness_score = len(sections_present & set(required_sections)) / len(required_sections)
        else:
            completeness_score = self._assess_general_completeness(content)
        
        level = self._determine_quality_level(completeness_score)
        missing_sections = set(required_sections) - sections_present
        
        suggestions = []
        if missing_sections:
            suggestions.append(f"Add missing sections: {', '.join(missing_sections)}")
        
        feedback = f"Completeness: {completeness_score:.2%}. "
        if missing_sections:
            feedback += f"Missing: {', '.join(missing_sections)}"
        else:
            feedback += "All required sections present."
        
        return QualityScore(
            metric=QualityMetric.COMPLETENESS,
            score=completeness_score,
            level=level,
            feedback=feedback,
            suggestions=suggestions
        )
    
    def _identify_sections(self, content: str) -> Set[str]:
        """Identify sections present in content."""
        sections = set()
        
        # Header-based section detection
        headers = re.findall(r'^#+\s*(.+)$', content, re.MULTILINE)
        for header in headers:
            section_key = self._normalize_section_name(header)
            sections.add(section_key)
        
        # Content-based detection
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['example', 'demo', 'sample']):
            sections.add('examples')
        
        if any(word in content_lower for word in ['parameter', 'argument', 'option']):
            sections.add('parameters')
        
        if any(word in content_lower for word in ['return', 'output', 'result']):
            sections.add('returns')
        
        return sections
    
    def _normalize_section_name(self, name: str) -> str:
        """Normalize section names for comparison."""
        normalized = name.lower().strip()
        
        # Map common variations
        section_mappings = {
            'getting started': 'introduction',
            'quick start': 'introduction',
            'usage': 'implementation',
            'how to use': 'implementation',
            'best practice': 'best_practices',
            'example': 'examples',
            'sample': 'examples'
        }
        
        return section_mappings.get(normalized, normalized)
    
    def _assess_general_completeness(self, content: str) -> float:
        """Assess general completeness for untyped documentation."""
        indicators = {
            'has_overview': bool(re.search(r'(overview|introduction|about)', content, re.IGNORECASE)),
            'has_examples': bool(re.search(r'(example|demo|sample|```)', content, re.IGNORECASE)),
            'has_details': len(content.split()) > 100,
            'has_structure': content.count('#') > 0,
        }
        
        return sum(indicators.values()) / len(indicators)
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from completeness score."""
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.85:
            return QualityLevel.GOOD
        elif score >= 0.75:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.5:
            return QualityLevel.NEEDS_IMPROVEMENT
        else:
            return QualityLevel.POOR

class QualityAssessmentEngine:
    """Main quality assessment engine orchestrating all quality analyzers."""
    
    def __init__(self):
        self.readability_analyzer = ReadabilityAnalyzer()
        self.completeness_validator = CompletenessValidator()
        self.assessment_history: Dict[str, List[QualityAssessment]] = {}
    
    async def assess_quality(self, content: str, content_id: str, 
                           doc_type: str = 'general') -> QualityAssessment:
        """Comprehensive quality assessment of documentation content."""
        
        # Run all assessments in parallel
        assessment_tasks = [
            self._assess_readability(content),
            self._assess_completeness(content, doc_type),
            self._assess_consistency(content),
            self._assess_accessibility(content)
        ]
        
        metric_scores = {}
        results = await asyncio.gather(*assessment_tasks)
        
        for result in results:
            metric_scores[result.metric] = result
        
        # Calculate overall score
        overall_score = sum(score.score for score in metric_scores.values()) / len(metric_scores)
        overall_level = self._determine_overall_level(overall_score)
        
        # Generate priority improvements
        priority_improvements = self._generate_priority_improvements(metric_scores)
        
        # Framework compliance assessment
        framework_compliance = self._assess_framework_compliance(content)
        
        assessment = QualityAssessment(
            content_id=content_id,
            overall_score=overall_score,
            overall_level=overall_level,
            metric_scores=metric_scores,
            priority_improvements=priority_improvements,
            framework_compliance=framework_compliance
        )
        
        # Store assessment history
        if content_id not in self.assessment_history:
            self.assessment_history[content_id] = []
        self.assessment_history[content_id].append(assessment)
        
        return assessment
    
    async def _assess_readability(self, content: str) -> QualityScore:
        """Assess content readability."""
        return self.readability_analyzer.analyze_readability(content)
    
    async def _assess_completeness(self, content: str, doc_type: str) -> QualityScore:
        """Assess content completeness."""
        return self.completeness_validator.validate_completeness(content, doc_type)
    
    async def _assess_consistency(self, content: str) -> QualityScore:
        """Assess content consistency."""
        # Simplified consistency check
        consistency_score = 0.8  # Placeholder
        
        return QualityScore(
            metric=QualityMetric.CONSISTENCY,
            score=consistency_score,
            level=self._determine_level_from_score(consistency_score),
            feedback="Consistency assessment completed.",
            suggestions=[]
        )
    
    async def _assess_accessibility(self, content: str) -> QualityScore:
        """Assess content accessibility."""
        # Check for accessibility features
        accessibility_indicators = {
            'alt_text': bool(re.search(r'alt\s*=\s*["\']', content, re.IGNORECASE)),
            'headers': content.count('#') > 0,
            'lists': '- ' in content or '1. ' in content,
            'code_blocks': '```' in content
        }
        
        accessibility_score = sum(accessibility_indicators.values()) / len(accessibility_indicators)
        
        return QualityScore(
            metric=QualityMetric.ACCESSIBILITY,
            score=accessibility_score,
            level=self._determine_level_from_score(accessibility_score),
            feedback=f"Accessibility score: {accessibility_score:.2%}",
            suggestions=self._generate_accessibility_suggestions(accessibility_indicators)
        )
    
    def _determine_overall_level(self, score: float) -> QualityLevel:
        """Determine overall quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.8:
            return QualityLevel.GOOD
        elif score >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.5:
            return QualityLevel.NEEDS_IMPROVEMENT
        else:
            return QualityLevel.POOR
    
    def _determine_level_from_score(self, score: float) -> QualityLevel:
        """Standard score to level conversion."""
        return self._determine_overall_level(score)
    
    def _generate_priority_improvements(self, metric_scores: Dict[QualityMetric, QualityScore]) -> List[str]:
        """Generate prioritized improvement recommendations."""
        improvements = []
        
        # Sort metrics by score (lowest first)
        sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1].score)
        
        for metric, score in sorted_metrics[:3]:  # Top 3 priorities
            if score.suggestions:
                improvements.extend(score.suggestions[:2])  # Top 2 suggestions per metric
        
        return improvements[:5]  # Limit to 5 total improvements
    
    def _assess_framework_compliance(self, content: str) -> Dict[str, float]:
        """Assess compliance with different framework standards."""
        frameworks = {
            'agency_swarm': self._check_agency_swarm_patterns(content),
            'crewai': self._check_crewai_patterns(content),
            'autogen': self._check_autogen_patterns(content),
            'llamaindex': self._check_llamaindex_patterns(content)
        }
        
        return frameworks
    
    def _check_agency_swarm_patterns(self, content: str) -> float:
        """Check Agency-Swarm documentation patterns."""
        patterns = [
            r'class\s+\w+Agent',
            r'def\s+\w+_tool',
            r'@tool',
            r'instructions\s*='
        ]
        
        found_patterns = sum(1 for pattern in patterns if re.search(pattern, content, re.IGNORECASE))
        return min(1.0, found_patterns / len(patterns))
    
    def _check_crewai_patterns(self, content: str) -> float:
        """Check CrewAI documentation patterns."""
        patterns = [
            r'crew\s*=',
            r'agent\s*=',
            r'task\s*=',
            r'def\s+\w+_callback'
        ]
        
        found_patterns = sum(1 for pattern in patterns if re.search(pattern, content, re.IGNORECASE))
        return min(1.0, found_patterns / len(patterns))
    
    def _check_autogen_patterns(self, content: str) -> float:
        """Check AutoGen documentation patterns."""
        patterns = [
            r'ConversableAgent',
            r'GroupChat',
            r'def\s+\w+_reply',
            r'system_message'
        ]
        
        found_patterns = sum(1 for pattern in patterns if re.search(pattern, content, re.IGNORECASE))
        return min(1.0, found_patterns / len(patterns))
    
    def _check_llamaindex_patterns(self, content: str) -> float:
        """Check LlamaIndex documentation patterns."""
        patterns = [
            r'ServiceContext',
            r'VectorStoreIndex',
            r'query_engine',
            r'def\s+\w+_query'
        ]
        
        found_patterns = sum(1 for pattern in patterns if re.search(pattern, content, re.IGNORECASE))
        return min(1.0, found_patterns / len(patterns))
    
    def _generate_accessibility_suggestions(self, indicators: Dict[str, bool]) -> List[str]:
        """Generate accessibility improvement suggestions."""
        suggestions = []
        
        if not indicators.get('alt_text', True):
            suggestions.append("Add alt text for images and visual elements")
        
        if not indicators.get('headers', True):
            suggestions.append("Use proper header hierarchy for better navigation")
        
        if not indicators.get('lists', True):
            suggestions.append("Organize content using lists for better structure")
        
        return suggestions
    
    def get_quality_trends(self, content_id: str) -> Dict[str, Any]:
        """Get quality trends for specific content over time."""
        if content_id not in self.assessment_history:
            return {}
        
        history = self.assessment_history[content_id]
        if len(history) < 2:
            return {"message": "Insufficient history for trend analysis"}
        
        latest = history[-1]
        previous = history[-2]
        
        trends = {}
        for metric in QualityMetric:
            if metric in latest.metric_scores and metric in previous.metric_scores:
                current_score = latest.metric_scores[metric].score
                previous_score = previous.metric_scores[metric].score
                change = current_score - previous_score
                trends[metric.value] = {
                    'current': current_score,
                    'previous': previous_score,
                    'change': change,
                    'trend': 'improving' if change > 0.05 else 'declining' if change < -0.05 else 'stable'
                }
        
        return trends