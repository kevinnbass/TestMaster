"""
Documentation Intelligence Analyzer

Core analysis engine for documentation intelligence insights.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re
import ast
from collections import defaultdict, Counter

from .metrics import (
    DocumentationType, IntelligenceMetric, OptimizationPriority,
    DocumentationMetrics, OptimizationRecommendation, IntelligenceInsight,
    TrendAnalysis, MetricsCalculator
)

logger = logging.getLogger(__name__)


class DocumentationAnalyzer:
    """
    Core analyzer for documentation intelligence.
    Provides deep analysis and insights for documentation quality.
    """
    
    def __init__(self):
        """Initialize the documentation analyzer."""
        try:
            self.metrics_calculator = MetricsCalculator()
            self.analysis_cache = {}
            logger.info("Documentation Analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")
            raise
    
    async def analyze_document(self, 
                             document_path: Path, 
                             content: str,
                             document_type: DocumentationType,
                             context: Optional[Dict[str, Any]] = None) -> DocumentationMetrics:
        """
        Perform comprehensive analysis of a documentation document.
        
        Args:
            document_path: Path to the document
            content: Document content
            document_type: Type of documentation
            context: Additional context for analysis
            
        Returns:
            Comprehensive metrics for the document
        """
        try:
            document_id = str(document_path)
            
            # Check cache first
            cache_key = f"{document_id}_{hash(content)}"
            if cache_key in self.analysis_cache:
                logger.debug(f"Using cached analysis for {document_path}")
                return self.analysis_cache[cache_key]
            
            # Perform comprehensive analysis
            metrics = await self._perform_deep_analysis(
                document_id, content, document_type, context or {}
            )
            
            # Cache results
            self.analysis_cache[cache_key] = metrics
            
            logger.info(f"Analyzed document: {document_path} (score: {metrics.calculate_overall_score():.1f})")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to analyze document {document_path}: {e}")
            # Return default metrics on error
            return DocumentationMetrics(
                document_id=str(document_path),
                document_type=document_type,
                readability_score=0.0,
                completeness_index=0.0
            )
    
    async def _perform_deep_analysis(self,
                                   document_id: str,
                                   content: str,
                                   document_type: DocumentationType,
                                   context: Dict[str, Any]) -> DocumentationMetrics:
        """Perform deep analysis of document content."""
        try:
            # Basic content metrics
            word_count = len(content.split())
            section_count = self._count_sections(content)
            code_example_count = self._count_code_examples(content)
            diagram_count = self._count_diagrams(content)
            external_link_count = self._count_external_links(content)
            
            # Calculate quality metrics
            readability_score = self.metrics_calculator.calculate_readability_score(content)
            
            # Get required sections based on document type
            required_sections = self._get_required_sections(document_type)
            completeness_index = self.metrics_calculator.calculate_completeness_index(
                content, required_sections
            )
            
            consistency_score = self.metrics_calculator.calculate_consistency_score(content)
            
            # Calculate advanced metrics
            accuracy_rating = await self._calculate_accuracy_rating(content, context)
            usefulness_index = await self._calculate_usefulness_index(content, document_type)
            maintenance_burden = self._calculate_maintenance_burden(content, context)
            
            # Technical metrics
            api_coverage = self._calculate_api_coverage(content, context)
            code_coverage = self._calculate_code_coverage(content, context)
            example_accuracy = self._calculate_example_accuracy(content)
            
            return DocumentationMetrics(
                document_id=document_id,
                document_type=document_type,
                readability_score=readability_score,
                completeness_index=completeness_index,
                accuracy_rating=accuracy_rating,
                consistency_score=consistency_score,
                usefulness_index=usefulness_index,
                maintenance_burden=maintenance_burden,
                word_count=word_count,
                section_count=section_count,
                code_example_count=code_example_count,
                diagram_count=diagram_count,
                external_link_count=external_link_count,
                api_coverage=api_coverage,
                code_coverage=code_coverage,
                example_accuracy=example_accuracy,
                last_updated=context.get('last_updated'),
                author_count=context.get('author_count', 1),
                review_count=context.get('review_count', 0)
            )
            
        except Exception as e:
            logger.error(f"Error in deep analysis: {e}")
            raise
    
    def _count_sections(self, content: str) -> int:
        """Count sections in the document."""
        try:
            # Count markdown headers
            headers = re.findall(r'^#+\s+', content, re.MULTILINE)
            
            # Count numbered sections
            numbered = re.findall(r'^\d+\.\s+', content, re.MULTILINE)
            
            # Count title case headers
            title_headers = re.findall(r'^[A-Z][A-Za-z\s]+:$', content, re.MULTILINE)
            
            return len(headers) + len(numbered) + len(title_headers)
        except Exception as e:
            logger.error(f"Error counting sections: {e}")
            return 0
    
    def _count_code_examples(self, content: str) -> int:
        """Count code examples in the document."""
        try:
            # Count code blocks
            code_blocks = len(re.findall(r'```[\s\S]*?```', content))
            
            # Count inline code
            inline_code = len(re.findall(r'`[^`]+`', content))
            
            # Count indented code (4+ spaces)
            indented_code = len(re.findall(r'^    [^\s].*$', content, re.MULTILINE))
            
            return code_blocks + (inline_code // 5) + (indented_code // 3)  # Normalize counts
        except Exception as e:
            logger.error(f"Error counting code examples: {e}")
            return 0
    
    def _count_diagrams(self, content: str) -> int:
        """Count diagrams and visual elements."""
        try:
            # Count image references
            images = len(re.findall(r'!\[.*?\]\(.*?\)', content))
            
            # Count diagram indicators
            diagram_keywords = ['mermaid', 'plantuml', 'diagram', 'flowchart', 'sequence']
            diagram_count = sum(content.lower().count(keyword) for keyword in diagram_keywords)
            
            return images + diagram_count
        except Exception as e:
            logger.error(f"Error counting diagrams: {e}")
            return 0
    
    def _count_external_links(self, content: str) -> int:
        """Count external links."""
        try:
            # Count markdown links
            links = re.findall(r'\[.*?\]\((https?://.*?)\)', content)
            
            # Count bare URLs
            bare_urls = re.findall(r'https?://[^\s\)]+', content)
            
            return len(links) + len(bare_urls)
        except Exception as e:
            logger.error(f"Error counting external links: {e}")
            return 0
    
    def _get_required_sections(self, document_type: DocumentationType) -> List[str]:
        """Get required sections based on document type."""
        try:
            section_map = {
                DocumentationType.API_DOCUMENTATION: [
                    'overview', 'authentication', 'endpoints', 'examples', 'errors'
                ],
                DocumentationType.CODE_DOCUMENTATION: [
                    'description', 'parameters', 'returns', 'examples', 'usage'
                ],
                DocumentationType.ARCHITECTURE_DOCS: [
                    'overview', 'components', 'data flow', 'deployment', 'monitoring'
                ],
                DocumentationType.USER_GUIDES: [
                    'introduction', 'getting started', 'features', 'troubleshooting'
                ],
                DocumentationType.TECHNICAL_SPECS: [
                    'requirements', 'design', 'implementation', 'testing', 'deployment'
                ],
                DocumentationType.COMPLIANCE_DOCS: [
                    'scope', 'requirements', 'controls', 'assessment', 'monitoring'
                ],
                DocumentationType.SECURITY_DOCS: [
                    'threats', 'controls', 'procedures', 'monitoring', 'incident response'
                ]
            }
            
            return section_map.get(document_type, ['overview', 'description', 'usage'])
        except Exception as e:
            logger.error(f"Error getting required sections: {e}")
            return []
    
    async def _calculate_accuracy_rating(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate accuracy rating based on content validation."""
        try:
            base_score = 85.0  # Start with good baseline
            
            # Check for outdated information indicators
            outdated_indicators = [
                'deprecated', 'obsolete', 'legacy', 'old version', 'previous version'
            ]
            
            content_lower = content.lower()
            outdated_count = sum(content_lower.count(indicator) for indicator in outdated_indicators)
            
            # Penalty for outdated content
            accuracy_penalty = min(outdated_count * 10, 30)  # Max 30 point penalty
            
            # Bonus for recent updates
            last_updated = context.get('last_updated')
            if last_updated:
                try:
                    update_date = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    days_since_update = (datetime.utcnow() - update_date.replace(tzinfo=None)).days
                    
                    if days_since_update < 30:
                        base_score += 10  # Recent update bonus
                    elif days_since_update > 365:
                        base_score -= 15  # Old content penalty
                except Exception:
                    pass  # Ignore date parsing errors
            
            return max(0.0, min(100.0, base_score - accuracy_penalty))
            
        except Exception as e:
            logger.error(f"Error calculating accuracy rating: {e}")
            return 75.0  # Default score on error
    
    async def _calculate_usefulness_index(self, content: str, document_type: DocumentationType) -> float:
        """Calculate usefulness index based on content analysis."""
        try:
            base_score = 70.0
            
            # Useful content indicators
            useful_indicators = {
                'examples': 15,
                'tutorial': 10,
                'step-by-step': 10,
                'troubleshooting': 12,
                'best practices': 8,
                'common issues': 8,
                'frequently asked': 5
            }
            
            content_lower = content.lower()
            usefulness_score = 0
            
            for indicator, score in useful_indicators.items():
                if indicator in content_lower:
                    usefulness_score += score
            
            # Cap the bonus
            usefulness_bonus = min(usefulness_score, 25)
            
            return min(100.0, base_score + usefulness_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating usefulness index: {e}")
            return 70.0
    
    def _calculate_maintenance_burden(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate maintenance burden score."""
        try:
            burden_score = 0.0
            
            # High maintenance indicators
            maintenance_indicators = {
                'hardcoded': 10,
                'specific version': 8,
                'exact date': 5,
                'temporary': 12,
                'TODO': 15,
                'FIXME': 15,
                'hack': 20,
                'workaround': 8
            }
            
            content_lower = content.lower()
            
            for indicator, burden in maintenance_indicators.items():
                count = content_lower.count(indicator)
                burden_score += count * burden
            
            # Factor in document size (larger docs are harder to maintain)
            word_count = len(content.split())
            if word_count > 5000:
                burden_score += 20
            elif word_count > 2000:
                burden_score += 10
            
            return min(100.0, burden_score)
            
        except Exception as e:
            logger.error(f"Error calculating maintenance burden: {e}")
            return 0.0
    
    def _calculate_api_coverage(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate API coverage percentage."""
        try:
            # This would ideally integrate with actual API discovery
            # For now, use heuristics based on content
            
            api_indicators = ['endpoint', 'method', 'parameter', 'response', 'request']
            content_lower = content.lower()
            
            indicator_count = sum(content_lower.count(indicator) for indicator in api_indicators)
            
            # Estimate coverage based on indicator density
            word_count = len(content.split())
            if word_count == 0:
                return 0.0
            
            density = indicator_count / word_count * 1000  # Per 1000 words
            
            # Convert density to coverage estimate
            coverage = min(density * 10, 100.0)  # Max 100%
            
            return coverage
            
        except Exception as e:
            logger.error(f"Error calculating API coverage: {e}")
            return 0.0
    
    def _calculate_code_coverage(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate code coverage in documentation."""
        try:
            # Count code elements
            code_blocks = len(re.findall(r'```[\s\S]*?```', content))
            inline_code = len(re.findall(r'`[^`]+`', content))
            
            total_code_elements = code_blocks + inline_code
            
            # Estimate coverage based on content length and code density
            word_count = len(content.split())
            if word_count == 0:
                return 0.0
            
            # Heuristic: good documentation has code examples for every 200-300 words
            expected_code_elements = word_count / 250
            
            if expected_code_elements == 0:
                return 100.0 if total_code_elements == 0 else 0.0
            
            coverage = min((total_code_elements / expected_code_elements) * 100, 100.0)
            
            return coverage
            
        except Exception as e:
            logger.error(f"Error calculating code coverage: {e}")
            return 0.0
    
    def _calculate_example_accuracy(self, content: str) -> float:
        """Calculate accuracy of code examples."""
        try:
            # Extract code blocks
            code_blocks = re.findall(r'```(?:python|javascript|bash|shell|json)?\n([\s\S]*?)```', content)
            
            if not code_blocks:
                return 100.0  # No examples to evaluate
            
            accurate_examples = 0
            total_examples = len(code_blocks)
            
            for code_block in code_blocks:
                try:
                    # Basic syntax validation for Python code
                    if 'import' in code_block or 'def ' in code_block or 'class ' in code_block:
                        # Try to parse as Python
                        ast.parse(code_block)
                        accurate_examples += 1
                    else:
                        # For non-Python code, do basic checks
                        if len(code_block.strip()) > 0 and not code_block.strip().startswith('#'):
                            accurate_examples += 1
                except SyntaxError:
                    # Syntax error in code example
                    pass
                except Exception:
                    # Other parsing issues, assume it's valid
                    accurate_examples += 1
            
            return (accurate_examples / total_examples) * 100.0 if total_examples > 0 else 100.0
            
        except Exception as e:
            logger.error(f"Error calculating example accuracy: {e}")
            return 85.0  # Default to reasonable score