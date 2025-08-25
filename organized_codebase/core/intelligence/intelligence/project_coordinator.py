"""
Documentation Intelligence Project Coordinator

Handles project-level analysis, coordination, and reporting for documentation intelligence.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from .metrics import DocumentationType, DocumentationMetrics, OptimizationRecommendation, IntelligenceInsight
from .analyzer import DocumentationAnalyzer
from .optimizer import DocumentationOptimizer

logger = logging.getLogger(__name__)


class ProjectCoordinator:
    """
    Coordinates project-level documentation intelligence operations.
    Manages analysis workflows, improvement planning, and progress tracking.
    """
    
    def __init__(self, analyzer: DocumentationAnalyzer, optimizer: DocumentationOptimizer):
        """
        Initialize project coordinator.
        
        Args:
            analyzer: Documentation analyzer instance
            optimizer: Documentation optimizer instance
        """
        try:
            self.analyzer = analyzer
            self.optimizer = optimizer
            self.batch_processing_enabled = True
            logger.info("Project Coordinator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize project coordinator: {e}")
            raise
    
    async def analyze_project(self, 
                            project_path: Path,
                            document_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive project-wide documentation analysis.
        
        Args:
            project_path: Path to project root
            document_patterns: File patterns to include
            
        Returns:
            Comprehensive project analysis report
        """
        try:
            document_patterns = document_patterns or ['*.md', '*.rst', '*.txt']
            
            # Discover documents
            documents = []
            for pattern in document_patterns:
                documents.extend(project_path.rglob(pattern))
            
            if not documents:
                logger.warning(f"No documents found in {project_path}")
                return self._empty_project_report(project_path)
            
            # Analyze documents in batches
            all_metrics = await self._analyze_document_batch(documents)
            
            # Generate project-level insights and recommendations
            project_insights = await self._generate_project_insights(all_metrics)
            project_recommendations = await self._generate_project_recommendations(all_metrics)
            project_stats = self._calculate_project_statistics(all_metrics)
            
            report = {
                'project_path': str(project_path),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'documents_analyzed': len(all_metrics),
                'project_statistics': project_stats,
                'document_metrics': [self._metrics_to_dict(m) for m in all_metrics],
                'project_insights': [self._insight_to_dict(i) for i in project_insights],
                'optimization_recommendations': [self._recommendation_to_dict(r) for r in project_recommendations],
                'quality_summary': self._generate_quality_summary(all_metrics)
            }
            
            logger.info(f"Analyzed {len(documents)} documents in project {project_path.name}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to analyze project {project_path}: {e}")
            return self._error_project_report(project_path, str(e))
    
    async def generate_improvement_plan(self, 
                                      analysis_results: List[Dict[str, Any]],
                                      timeline_weeks: int = 12) -> Dict[str, Any]:
        """
        Generate comprehensive improvement plan based on analysis results.
        
        Args:
            analysis_results: List of analysis results from documents/projects
            timeline_weeks: Timeline for improvement plan in weeks
            
        Returns:
            Detailed improvement plan
        """
        try:
            all_recommendations = []
            
            # Collect all recommendations
            for result in analysis_results:
                if 'recommendations' in result:
                    all_recommendations.extend(result['recommendations'])
                if 'optimization_recommendations' in result:
                    all_recommendations.extend(result['optimization_recommendations'])
            
            if not all_recommendations:
                return self._empty_improvement_plan()
            
            # Prioritize and organize recommendations
            prioritized_recommendations = self._prioritize_recommendations(all_recommendations)
            
            # Create implementation plan
            improvement_plan = {
                'plan_id': f"improvement_plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'created_at': datetime.utcnow().isoformat(),
                'timeline_weeks': timeline_weeks,
                'total_recommendations': len(all_recommendations),
                'prioritized_recommendations': prioritized_recommendations,
                'implementation_timeline': self._create_improvement_timeline(prioritized_recommendations, timeline_weeks),
                'resource_requirements': self._calculate_resource_requirements(prioritized_recommendations),
                'success_metrics': self._define_success_metrics(prioritized_recommendations),
                'expected_outcomes': self._calculate_expected_outcomes(prioritized_recommendations),
                'risk_assessment': self._assess_implementation_risks(prioritized_recommendations)
            }
            
            logger.info(f"Generated improvement plan with {len(prioritized_recommendations)} recommendations")
            return improvement_plan
            
        except Exception as e:
            logger.error(f"Failed to generate improvement plan: {e}")
            return {'error': str(e), 'plan_id': 'error_plan'}
    
    async def track_progress(self, 
                           improvement_plan_id: str,
                           completed_items: List[str]) -> Dict[str, Any]:
        """
        Track progress on an improvement plan.
        
        Args:
            improvement_plan_id: ID of the improvement plan
            completed_items: List of completed recommendation IDs
            
        Returns:
            Progress tracking report
        """
        try:
            progress_report = {
                'plan_id': improvement_plan_id,
                'tracking_date': datetime.utcnow().isoformat(),
                'completed_items': len(completed_items),
                'completion_rate': self._calculate_completion_rate(completed_items),
                'current_phase': self._determine_current_phase(completed_items),
                'next_actions': self._get_next_actions(completed_items),
                'blockers': self._identify_blockers(completed_items),
                'recommendations': [
                    'Continue with planned activities',
                    'Monitor quality metrics',
                    'Review progress weekly'
                ]
            }
            
            logger.info(f"Tracked progress for plan {improvement_plan_id}")
            return progress_report
            
        except Exception as e:
            logger.error(f"Failed to track progress: {e}")
            return {'error': str(e), 'plan_id': improvement_plan_id}
    
    # Private helper methods
    async def _analyze_document_batch(self, documents: List[Path]) -> List[DocumentationMetrics]:
        """Analyze documents in batches for performance."""
        try:
            batch_size = 10 if self.batch_processing_enabled else 1
            all_metrics = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_metrics = await self._process_document_batch(batch)
                all_metrics.extend(batch_metrics)
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return []
    
    async def _process_document_batch(self, documents: List[Path]) -> List[DocumentationMetrics]:
        """Process a single batch of documents."""
        try:
            tasks = []
            for doc_path in documents:
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc_type = self._infer_document_type(doc_path, content)
                    task = self.analyzer.analyze_document(doc_path, content, doc_type)
                    tasks.append(task)
                    
                except Exception as e:
                    logger.warning(f"Skipping document {doc_path}: {e}")
                    continue
            
            if tasks:
                metrics_list = await asyncio.gather(*tasks, return_exceptions=True)
                return [m for m in metrics_list if isinstance(m, DocumentationMetrics)]
            
            return []
            
        except Exception as e:
            logger.error(f"Error processing document batch: {e}")
            return []
    
    def _infer_document_type(self, document_path: Path, content: str) -> DocumentationType:
        """Infer document type from path and content."""
        try:
            filename = document_path.name.lower()
            content_lower = content.lower()
            
            # Type inference logic
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
    
    async def _generate_project_insights(self, metrics_list: List[DocumentationMetrics]) -> List[IntelligenceInsight]:
        """Generate project-level insights."""
        try:
            if not metrics_list:
                return []
            
            insights = []
            avg_quality = sum(m.calculate_overall_score() for m in metrics_list) / len(metrics_list)
            
            if avg_quality >= 85:
                insights.append(IntelligenceInsight(
                    insight_id="project_quality_excellent",
                    insight_type="project_quality",
                    confidence=0.9,
                    title="Excellent Project Documentation Quality",
                    description=f"Project maintains high documentation quality with average score of {avg_quality:.1f}",
                    evidence=[f"Average quality score: {avg_quality:.1f}", f"Documents analyzed: {len(metrics_list)}"]
                ))
            elif avg_quality < 60:
                insights.append(IntelligenceInsight(
                    insight_id="project_quality_poor",
                    insight_type="project_quality",
                    confidence=0.95,
                    title="Project Documentation Needs Major Improvement",
                    description="Project documentation quality is below acceptable standards",
                    evidence=[f"Low average score: {avg_quality:.1f}", "Multiple documents need attention"],
                    actionable=True,
                    recommendations=[
                        "Implement documentation quality standards",
                        "Conduct comprehensive content audit",
                        "Establish regular review processes"
                    ]
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating project insights: {e}")
            return []
    
    async def _generate_project_recommendations(self, metrics_list: List[DocumentationMetrics]) -> List[OptimizationRecommendation]:
        """Generate project-level recommendations."""
        try:
            if not metrics_list:
                return []
            
            from .metrics import OptimizationPriority
            
            avg_completeness = sum(m.completeness_index for m in metrics_list) / len(metrics_list)
            
            if avg_completeness < 75:
                return [OptimizationRecommendation(
                    recommendation_id="project_completeness_improvement",
                    priority=OptimizationPriority.HIGH,
                    category="Project-wide",
                    title="Improve Project Documentation Completeness",
                    description="Multiple documents across the project lack required sections",
                    impact_score=80.0,
                    effort_estimate=20,
                    current_state=f"Average completeness: {avg_completeness:.1f}%",
                    target_state="Average completeness: 85%+",
                    action_items=[
                        "Establish documentation templates",
                        "Define required sections per document type",
                        "Implement automated completeness checks",
                        "Train team on documentation standards"
                    ],
                    affected_sections=["All project documentation"],
                    stakeholder_groups=["Documentation team", "Development team"],
                    expected_improvement={"project_completeness": 15.0, "team_productivity": 25.0}
                )]
            
            return []
            
        except Exception as e:
            logger.error(f"Error generating project recommendations: {e}")
            return []
    
    def _calculate_project_statistics(self, metrics_list: List[DocumentationMetrics]) -> Dict[str, Any]:
        """Calculate project-level statistics."""
        try:
            if not metrics_list:
                return {}
            
            quality_scores = [m.calculate_overall_score() for m in metrics_list]
            
            return {
                'total_documents': len(metrics_list),
                'average_quality_score': sum(quality_scores) / len(quality_scores),
                'highest_quality_score': max(quality_scores),
                'lowest_quality_score': min(quality_scores),
                'documents_above_80': len([s for s in quality_scores if s >= 80]),
                'documents_below_60': len([s for s in quality_scores if s < 60]),
                'total_word_count': sum(m.word_count for m in metrics_list),
                'total_code_examples': sum(m.code_example_count for m in metrics_list),
                'average_readability': sum(m.readability_score for m in metrics_list) / len(metrics_list),
                'average_completeness': sum(m.completeness_index for m in metrics_list) / len(metrics_list)
            }
            
        except Exception as e:
            logger.error(f"Error calculating project statistics: {e}")
            return {}
    
    def _generate_quality_summary(self, metrics_list: List[DocumentationMetrics]) -> Dict[str, Any]:
        """Generate quality summary for the project."""
        try:
            if not metrics_list:
                return {'status': 'no_data'}
            
            avg_score = sum(m.calculate_overall_score() for m in metrics_list) / len(metrics_list)
            
            if avg_score >= 85:
                status, message = 'excellent', 'Project documentation quality is excellent'
            elif avg_score >= 70:
                status, message = 'good', 'Project documentation quality is good with room for improvement'
            elif avg_score >= 50:
                status, message = 'fair', 'Project documentation quality is fair and needs attention'
            else:
                status, message = 'poor', 'Project documentation quality is poor and requires immediate action'
            
            return {
                'status': status,
                'message': message,
                'average_score': avg_score,
                'recommendation': self._get_quality_recommendation(status)
            }
            
        except Exception as e:
            logger.error(f"Error generating quality summary: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_quality_recommendation(self, status: str) -> str:
        """Get quality recommendation based on status."""
        recommendations = {
            'excellent': 'Maintain current quality standards and continue regular reviews',
            'good': 'Focus on specific improvement areas identified in recommendations',
            'fair': 'Implement systematic improvement plan and quality standards',
            'poor': 'Urgent action required - conduct comprehensive documentation overhaul'
        }
        return recommendations.get(status, 'Review documentation and implement improvements')
    
    # Helper methods for data conversion
    def _metrics_to_dict(self, metrics: DocumentationMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        try:
            return {
                'document_id': metrics.document_id,
                'document_type': metrics.document_type.value,
                'analysis_timestamp': metrics.analysis_timestamp,
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
    
    # Error handling and utility methods
    def _empty_project_report(self, project_path: Path) -> Dict[str, Any]:
        """Generate empty project report."""
        return {
            'project_path': str(project_path),
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'documents_analyzed': 0,
            'error': 'No documents found',
            'recommendations': ['Add documentation files to the project']
        }
    
    def _error_project_report(self, project_path: Path, error: str) -> Dict[str, Any]:
        """Generate error project report."""
        return {
            'project_path': str(project_path),
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'error': error,
            'status': 'error'
        }
    
    def _empty_improvement_plan(self) -> Dict[str, Any]:
        """Generate empty improvement plan."""
        return {
            'plan_id': 'empty_plan',
            'created_at': datetime.utcnow().isoformat(),
            'message': 'No recommendations available',
            'total_recommendations': 0
        }
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize recommendations by impact and effort."""
        try:
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'optional': 4}
            
            def sort_key(rec):
                priority = priority_order.get(rec.get('priority', 'low'), 3)
                roi = rec.get('roi', 0)
                return (priority, -roi)
            
            return sorted(recommendations, key=sort_key)
            
        except Exception as e:
            logger.error(f"Error prioritizing recommendations: {e}")
            return recommendations
    
    def _create_improvement_timeline(self, recommendations: List[Dict[str, Any]], weeks: int) -> Dict[str, Any]:
        """Create implementation timeline."""
        try:
            phases = {
                'Phase 1 (Weeks 1-4)': [],
                'Phase 2 (Weeks 5-8)': [],
                'Phase 3 (Weeks 9-12)': []
            }
            
            for i, rec in enumerate(recommendations[:12]):
                phase_index = i % 3
                phase_name = list(phases.keys())[phase_index]
                phases[phase_name].append(rec['title'])
            
            return phases
            
        except Exception as e:
            logger.error(f"Error creating timeline: {e}")
            return {}
    
    def _calculate_resource_requirements(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource requirements."""
        try:
            total_effort = sum(rec.get('effort_estimate', 0) for rec in recommendations)
            
            return {
                'total_effort_hours': total_effort,
                'estimated_weeks': max(1, total_effort // 20),
                'team_size_recommendation': max(1, total_effort // 160),
                'skills_required': ['Technical writing', 'Content strategy', 'Documentation tools']
            }
            
        except Exception as e:
            logger.error(f"Error calculating resource requirements: {e}")
            return {}
    
    def _define_success_metrics(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """Define success metrics for the improvement plan."""
        return [
            'Average documentation quality score > 85',
            'All critical recommendations implemented',
            'User satisfaction with documentation > 4/5',
            'Documentation maintenance burden < 20%',
            'API coverage > 90%'
        ]
    
    def _calculate_expected_outcomes(self, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate expected outcomes from implementing recommendations."""
        try:
            total_impact = sum(rec.get('impact_score', 0) for rec in recommendations)
            
            return {
                'quality_improvement': min(total_impact * 0.3, 30.0),
                'user_satisfaction_increase': min(total_impact * 0.2, 20.0),
                'maintenance_reduction': min(total_impact * 0.1, 15.0),
                'team_productivity_increase': min(total_impact * 0.15, 25.0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating expected outcomes: {e}")
            return {}
    
    def _assess_implementation_risks(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """Assess risks in implementing the improvement plan."""
        return [
            'Resource constraints may delay implementation',
            'Team availability for documentation work',
            'Maintaining quality during rapid changes',
            'User adoption of new documentation standards'
        ]
    
    def _calculate_completion_rate(self, completed_items: List[str]) -> float:
        """Calculate completion rate."""
        return min(len(completed_items) * 10.0, 100.0)  # Simplified calculation
    
    def _determine_current_phase(self, completed_items: List[str]) -> str:
        """Determine current phase based on completed items."""
        if len(completed_items) < 3:
            return 'planning'
        elif len(completed_items) < 8:
            return 'implementation'
        else:
            return 'optimization'
    
    def _get_next_actions(self, completed_items: List[str]) -> List[str]:
        """Get next actions based on progress."""
        return ['Continue with next priority items', 'Monitor implementation progress']
    
    def _identify_blockers(self, completed_items: List[str]) -> List[str]:
        """Identify potential blockers."""
        return []  # Simplified - would contain actual blocker detection logic