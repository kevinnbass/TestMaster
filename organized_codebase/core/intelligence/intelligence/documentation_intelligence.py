"""
Streamlined Documentation Intelligence Engine

Enterprise documentation analytics orchestrator - now modularized for simplicity.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

# Import modular intelligence components
from ..intelligence import (
    DocumentationType, DocumentationMetrics, OptimizationRecommendation, 
    IntelligenceInsight, DocumentationAnalyzer, DocumentationOptimizer
)
from ..intelligence.project_coordinator import ProjectCoordinator

logger = logging.getLogger(__name__)


class DocumentationIntelligenceEngine:
    """
    Streamlined documentation intelligence orchestrator.
    Coordinates analysis, optimization, and reporting through modular components.
    """
    
    def __init__(self):
        """Initialize the documentation intelligence engine with modular components."""
        try:
            # Initialize core analysis components
            self.analyzer = DocumentationAnalyzer()
            self.optimizer = DocumentationOptimizer()
            self.project_coordinator = ProjectCoordinator(self.analyzer, self.optimizer)
            
            # Initialize caches and state
            self.metrics_history = defaultdict(list)
            self.insights_cache = {}
            
            logger.info("Documentation Intelligence Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize intelligence engine: {e}")
            raise
    
    # High-level project operations (delegate to project coordinator)
    async def analyze_project(self, 
                            project_path: Path,
                            document_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive project-wide documentation analysis.
        Delegates to project coordinator for complex orchestration.
        """
        try:
            return await self.project_coordinator.analyze_project(project_path, document_patterns)
        except Exception as e:
            logger.error(f"Failed to analyze project {project_path}: {e}")
            return self._error_project_report(project_path, str(e))
    
    async def generate_improvement_plan(self, 
                                      analysis_results: List[Dict[str, Any]],
                                      timeline_weeks: int = 12) -> Dict[str, Any]:
        """
        Generate comprehensive improvement plan.
        Delegates to project coordinator for complex planning.
        """
        try:
            return await self.project_coordinator.generate_improvement_plan(analysis_results, timeline_weeks)
        except Exception as e:
            logger.error(f"Failed to generate improvement plan: {e}")
            return {'error': str(e), 'plan_id': 'error_plan'}
    
    async def track_progress(self, 
                           improvement_plan_id: str,
                           completed_items: List[str]) -> Dict[str, Any]:
        """
        Track progress on improvement plan.
        Delegates to project coordinator for progress management.
        """
        try:
            return await self.project_coordinator.track_progress(improvement_plan_id, completed_items)
        except Exception as e:
            logger.error(f"Failed to track progress: {e}")
            return {'error': str(e), 'plan_id': improvement_plan_id}
    
    # Single document operations (direct component usage)
    async def analyze_single_document(self, 
                                    document_path: Path,
                                    content: Optional[str] = None,
                                    document_type: Optional[DocumentationType] = None) -> Dict[str, Any]:
        """
        Perform detailed analysis of a single document.
        Uses analyzer and optimizer components directly for efficiency.
        """
        try:
            # Read content if not provided
            if content is None:
                try:
                    with open(document_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"Failed to read document {document_path}: {e}")
                    return self._error_document_report(document_path, f"File read error: {e}")
            
            # Infer document type if not provided
            if document_type is None:
                document_type = self._infer_document_type(document_path, content)
            
            # Perform analysis using analyzer component
            metrics = await self.analyzer.analyze_document(document_path, content, document_type)
            
            # Store in history for trend analysis
            self.metrics_history[str(document_path)].append(metrics)
            
            # Generate recommendations using optimizer component
            recommendations = await self.optimizer.generate_recommendations(metrics)
            
            # Generate insights with historical data
            historical_data = self.metrics_history[str(document_path)][:-1]  # Exclude current
            insights = await self.optimizer.generate_insights(metrics, historical_data)
            
            # Create comprehensive report
            report = {
                'document_path': str(document_path),
                'document_type': document_type.value,
                'analysis_timestamp': metrics.analysis_timestamp,
                'metrics': self._metrics_to_dict(metrics),
                'quality_score': metrics.calculate_overall_score(),
                'recommendations': [self._recommendation_to_dict(r) for r in recommendations],
                'insights': [self._insight_to_dict(i) for i in insights],
                'trend_analysis': self._analyze_document_trends(str(document_path)),
                'actionable_items': self._extract_actionable_items(recommendations, insights)
            }
            
            logger.info(f"Analyzed document: {document_path.name} (score: {metrics.calculate_overall_score():.1f})")
            return report
            
        except Exception as e:
            logger.error(f"Failed to analyze document {document_path}: {e}")
            return self._error_document_report(document_path, str(e))
    
    # Batch operations (optimized for performance)
    async def analyze_document_batch(self, document_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Analyze multiple documents efficiently in batch.
        Optimized for performance with concurrent processing.
        """
        try:
            tasks = []
            for doc_path in document_paths:
                task = self.analyze_single_document(doc_path)
                tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                valid_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
                return valid_results
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to analyze document batch: {e}")
            return []
    
    # Quality assessment operations
    async def assess_quality_trends(self, document_id: str, time_window_days: int = 30) -> Dict[str, Any]:
        """
        Assess quality trends for a specific document over time.
        """
        try:
            history = self.metrics_history.get(document_id, [])
            
            if len(history) < 2:
                return {
                    'document_id': document_id,
                    'trend_status': 'insufficient_data',
                    'message': 'Need more historical data for trend analysis'
                }
            
            # Analyze quality trend
            recent_scores = [m.calculate_overall_score() for m in history[-10:]]  # Last 10 analyses
            
            if len(recent_scores) >= 2:
                trend = recent_scores[-1] - recent_scores[0]
                direction = 'improving' if trend > 5 else 'declining' if trend < -5 else 'stable'
                
                return {
                    'document_id': document_id,
                    'trend_direction': direction,
                    'score_change': trend,
                    'analysis_count': len(history),
                    'latest_score': recent_scores[-1],
                    'average_score': sum(recent_scores) / len(recent_scores),
                    'trend_status': 'analyzed'
                }
            
            return {
                'document_id': document_id,
                'trend_status': 'insufficient_data',
                'analysis_count': len(history)
            }
            
        except Exception as e:
            logger.error(f"Failed to assess quality trends: {e}")
            return {'document_id': document_id, 'error': str(e)}
    
    # Optimization operations
    async def generate_quick_recommendations(self, document_path: Path) -> List[Dict[str, Any]]:
        """
        Generate quick optimization recommendations for a document.
        Lightweight operation for fast feedback.
        """
        try:
            # Quick analysis without full metrics
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_type = self._infer_document_type(document_path, content)
            metrics = await self.analyzer.analyze_document(document_path, content, doc_type)
            recommendations = await self.optimizer.generate_recommendations(metrics)
            
            return [self._recommendation_to_dict(r) for r in recommendations[:5]]  # Top 5 recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate quick recommendations: {e}")
            return []
    
    # Utility and helper methods
    def get_cached_analysis(self, document_path: Path) -> Optional[Dict[str, Any]]:
        """Get cached analysis results if available."""
        try:
            document_id = str(document_path)
            history = self.metrics_history.get(document_id, [])
            
            if history:
                latest_metrics = history[-1]
                return {
                    'document_path': document_id,
                    'cached_metrics': self._metrics_to_dict(latest_metrics),
                    'cache_timestamp': latest_metrics.analysis_timestamp,
                    'quality_score': latest_metrics.calculate_overall_score()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached analysis: {e}")
            return None
    
    def clear_cache(self, document_path: Optional[Path] = None) -> bool:
        """Clear analysis cache for specific document or all documents."""
        try:
            if document_path:
                document_id = str(document_path)
                if document_id in self.metrics_history:
                    del self.metrics_history[document_id]
                    logger.info(f"Cleared cache for {document_path}")
            else:
                self.metrics_history.clear()
                self.insights_cache.clear()
                logger.info("Cleared all analysis cache")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system statistics and health information."""
        try:
            total_documents = len(self.metrics_history)
            total_analyses = sum(len(history) for history in self.metrics_history.values())
            
            return {
                'total_documents_analyzed': total_documents,
                'total_analyses_performed': total_analyses,
                'cache_size': len(self.insights_cache),
                'components_active': {
                    'analyzer': True,
                    'optimizer': True,
                    'project_coordinator': True
                },
                'system_health': 'healthy'
            }
            
        except Exception as e:
            logger.error(f"Failed to get system statistics: {e}")
            return {
                'system_health': 'error',
                'error': str(e)
            }
    
    # Private helper methods
    def _infer_document_type(self, document_path: Path, content: str) -> DocumentationType:
        """Infer document type from path and content."""
        try:
            filename = document_path.name.lower()
            content_lower = content.lower()
            
            if any(indicator in content_lower for indicator in ['api', 'endpoint', 'rest', 'graphql']):
                return DocumentationType.API_DOCUMENTATION
            elif any(indicator in filename for indicator in ['architecture', 'design', 'system']):
                return DocumentationType.ARCHITECTURE_DOCS
            elif any(indicator in filename for indicator in ['guide', 'tutorial', 'manual', 'howto']):
                return DocumentationType.USER_GUIDES
            elif any(indicator in content_lower for indicator in ['security', 'authentication', 'authorization']):
                return DocumentationType.SECURITY_DOCS
            elif any(indicator in content_lower for indicator in ['compliance', 'gdpr', 'hipaa', 'sox']):
                return DocumentationType.COMPLIANCE_DOCS
            elif any(indicator in filename for indicator in ['spec', 'specification', 'requirements']):
                return DocumentationType.TECHNICAL_SPECS
            else:
                return DocumentationType.CODE_DOCUMENTATION
                
        except Exception as e:
            logger.error(f"Error inferring document type: {e}")
            return DocumentationType.CODE_DOCUMENTATION
    
    def _analyze_document_trends(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Analyze trends for a specific document."""
        try:
            history = self.metrics_history.get(document_id, [])
            
            if len(history) < 2:
                return None
            
            recent_scores = [m.calculate_overall_score() for m in history[-5:]]
            
            if len(recent_scores) >= 2:
                trend = recent_scores[-1] - recent_scores[0]
                direction = 'improving' if trend > 5 else 'declining' if trend < -5 else 'stable'
                
                return {
                    'trend_direction': direction,
                    'score_change': trend,
                    'analysis_count': len(history),
                    'latest_score': recent_scores[-1]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing document trends: {e}")
            return None
    
    def _extract_actionable_items(self, recommendations: List[OptimizationRecommendation], insights: List[IntelligenceInsight]) -> List[str]:
        """Extract actionable items from recommendations and insights."""
        try:
            actionable_items = []
            
            # Extract from recommendations
            for rec in recommendations:
                actionable_items.extend(rec.action_items)
            
            # Extract from actionable insights
            for insight in insights:
                if insight.actionable:
                    actionable_items.extend(insight.recommendations)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_items = []
            for item in actionable_items:
                if item not in seen:
                    seen.add(item)
                    unique_items.append(item)
            
            return unique_items[:10]  # Limit to top 10 items
            
        except Exception as e:
            logger.error(f"Error extracting actionable items: {e}")
            return []
    
    # Data conversion helpers
    def _metrics_to_dict(self, metrics: DocumentationMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        try:
            return {
                'document_id': metrics.document_id,
                'document_type': metrics.document_type.value,
                'quality_score': metrics.calculate_overall_score(),
                'readability_score': metrics.readability_score,
                'completeness_index': metrics.completeness_index,
                'accuracy_rating': metrics.accuracy_rating,
                'consistency_score': metrics.consistency_score,
                'usefulness_index': metrics.usefulness_index,
                'maintenance_burden': metrics.maintenance_burden,
                'word_count': metrics.word_count,
                'section_count': metrics.section_count,
                'code_example_count': metrics.code_example_count
            }
        except Exception as e:
            logger.error(f"Error converting metrics to dict: {e}")
            return {}
    
    def _insight_to_dict(self, insight: IntelligenceInsight) -> Dict[str, Any]:
        """Convert insight to dictionary."""
        try:
            return {
                'insight_id': insight.insight_id,
                'insight_type': insight.insight_type,
                'confidence': insight.confidence,
                'title': insight.title,
                'description': insight.description,
                'evidence': insight.evidence,
                'actionable': insight.actionable,
                'recommendations': insight.recommendations
            }
        except Exception as e:
            logger.error(f"Error converting insight to dict: {e}")
            return {}
    
    def _recommendation_to_dict(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        try:
            return {
                'recommendation_id': recommendation.recommendation_id,
                'priority': recommendation.priority.value,
                'category': recommendation.category,
                'title': recommendation.title,
                'description': recommendation.description,
                'impact_score': recommendation.impact_score,
                'effort_estimate': recommendation.effort_estimate,
                'action_items': recommendation.action_items,
                'expected_improvement': recommendation.expected_improvement,
                'roi': recommendation.calculate_roi()
            }
        except Exception as e:
            logger.error(f"Error converting recommendation to dict: {e}")
            return {}
    
    # Error handling methods
    def _error_project_report(self, project_path: Path, error: str) -> Dict[str, Any]:
        """Generate error project report."""
        return {
            'project_path': str(project_path),
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'error': error,
            'status': 'error'
        }
    
    def _error_document_report(self, document_path: Path, error: str) -> Dict[str, Any]:
        """Generate error document report."""
        return {
            'document_path': str(document_path),
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'error': error,
            'status': 'error'
        }