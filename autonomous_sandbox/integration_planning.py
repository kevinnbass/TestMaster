#!/usr/bin/env python3
"""
Intelligence Integration Engine Planning Module
===============================================

Planning and reorganization logic for the intelligence integration system.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime

from integration_models import ReorganizationRecommendation


class IntelligencePlanningEngine:
    """Handles reorganization planning and risk assessment"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the planning engine"""
        self.config = config

    def generate_reorganization_plan(self, integrated_intelligence: List,
                                   llm_intelligence_map: Dict[str, Any],
                                   reorganization_phases_class) -> Any:
        """Generate a comprehensive reorganization plan"""

        # Group by classification
        by_classification = defaultdict(list)
        for intelligence in integrated_intelligence:
            by_classification[intelligence.integrated_classification].append(intelligence)

        # Calculate statistics
        total_modules = len(integrated_intelligence)
        high_priority = len([i for i in integrated_intelligence if i.reorganization_priority >= 8])
        security_modules = len([i for i in integrated_intelligence if 'security' in i.integrated_classification])

        # Generate phases
        reorganization_phases = self._generate_reorganization_phases(integrated_intelligence, by_classification, reorganization_phases_class)

        # Calculate estimated effort
        estimated_hours = sum(phase.estimated_time_minutes for phase in reorganization_phases) / 60

        # Assess risks
        risk_assessment = self._assess_reorganization_risks(integrated_intelligence)

        # Define success metrics
        success_metrics = {
            'structural_metrics': [
                'Achieve logical grouping by functionality',
                'Reduce cognitive load for developers',
                'Improve separation of concerns',
                'Maintain import relationships'
            ],
            'quality_metrics': [
                'Maintain or improve code quality scores',
                'Reduce complexity in reorganized modules',
                'Better test coverage alignment',
                'Preserve security properties'
            ],
            'functional_metrics': [
                'All imports still work correctly',
                'No broken functionality',
                'Improved developer experience',
                'Enhanced maintainability'
            ],
            'target_goals': [
                f'Process {total_modules} modules',
                f'Handle {security_modules} security-related modules carefully',
                f'Complete high-priority items ({high_priority}) first',
                'Generate actionable reorganization roadmap'
            ]
        }

        # Implementation guidelines
        implementation_guidelines = [
            "Start with Phase 1 (high-confidence, low-risk moves) to build momentum",
            "Perform security review for Phase 2 modules before any moves",
            "Run comprehensive tests after each phase",
            "Monitor import dependencies and fix any broken references immediately",
            "Consider gradual rollout with feature flags if available",
            "Document any architectural decisions made during reorganization",
            "Maintain backup of original structure during transition",
            "Communicate changes to development team early"
        ]

        # Create plan object (would need the actual ReorganizationPlan class)
        plan = {
            'plan_timestamp': datetime.now().isoformat(),
            'total_modules': total_modules,
            'reorganization_phases': reorganization_phases,
            'estimated_total_time_hours': estimated_hours,
            'risk_assessment': risk_assessment,
            'success_metrics': success_metrics,
            'implementation_guidelines': implementation_guidelines
        }

        return plan

    def _generate_reorganization_phases(self, integrated_intelligence: List,
                                      by_classification: Dict[str, List],
                                      reorganization_phases_class) -> List:
        """Generate phased reorganization approach"""

        phases = []

        # Sort by priority
        sorted_intelligence = sorted(integrated_intelligence,
                                   key=lambda x: x.reorganization_priority,
                                   reverse=True)

        # Phase 1: High confidence, low risk
        phase1_modules = []
        for intelligence in sorted_intelligence:
            if (intelligence.integration_confidence > 0.8 and
                intelligence.reorganization_priority < 7 and
                'security' not in intelligence.integrated_classification):

                move_recommendations = [
                    rec for rec in intelligence.final_recommendations
                    if 'move to' in rec.lower() or 'reorganize' in rec.lower()
                ]

                if move_recommendations:
                    phase1_modules.append({
                        'file': intelligence.file_path,
                        'relative_path': intelligence.relative_path,
                        'target_classification': intelligence.integrated_classification,
                        'confidence': intelligence.integration_confidence,
                        'priority': intelligence.reorganization_priority,
                        'reasoning': intelligence.synthesis_reasoning,
                        'recommendations': move_recommendations
                    })

        if phase1_modules:
            phases.append(reorganization_phases_class(
                phase_number=1,
                phase_name='High Confidence Reorganization',
                description='Move modules with high analysis confidence and low risk',
                modules=phase1_modules,
                estimated_time_minutes=len(phase1_modules) * 15,
                risk_level='Low',
                prerequisites=['Backup current structure', 'Run test suite'],
                success_criteria=['All imports work correctly', 'No functionality broken', 'Tests pass']
            ))

        # Phase 2: Security-critical modules
        phase2_modules = []
        for intelligence in sorted_intelligence:
            if ('security' in intelligence.integrated_classification or
                'security' in intelligence.llm_analysis.security_implications.lower()):

                phase2_modules.append({
                    'file': intelligence.file_path,
                    'relative_path': intelligence.relative_path,
                    'classification': intelligence.integrated_classification,
                    'security_notes': intelligence.llm_analysis.security_implications,
                    'priority': intelligence.reorganization_priority,
                    'confidence': intelligence.integration_confidence
                })

        if phase2_modules:
            phases.append(reorganization_phases_class(
                phase_number=2,
                phase_name='Security-Critical Modules',
                description='Handle security-related modules with extra care and review',
                modules=phase2_modules,
                estimated_time_minutes=len(phase2_modules) * 30,
                risk_level='Medium',
                prerequisites=['Security review approval', 'Security testing', 'Backup critical files'],
                success_criteria=['Security properties preserved', 'No new vulnerabilities introduced', 'Security tests pass']
            ))

        # Phase 3: Complex and high-priority modules
        phase3_modules = []
        for intelligence in sorted_intelligence:
            if intelligence.reorganization_priority >= 8:

                phase3_modules.append({
                    'file': intelligence.file_path,
                    'relative_path': intelligence.relative_path,
                    'classification': intelligence.integrated_classification,
                    'complexity': intelligence.llm_analysis.complexity_assessment,
                    'priority': intelligence.reorganization_priority,
                    'confidence': intelligence.integration_confidence,
                    'issues': intelligence.llm_analysis.maintainability_notes
                })

        if phase3_modules:
            phases.append(reorganization_phases_class(
                phase_number=3,
                phase_name='Complex Module Reorganization',
                description='Handle complex, high-priority modules requiring careful planning',
                modules=phase3_modules,
                estimated_time_minutes=len(phase3_modules) * 45,
                risk_level='High',
                prerequisites=['Architecture review', 'Detailed planning', 'Team alignment'],
                success_criteria=['Complex dependencies resolved', 'Architecture improved', 'Maintainability enhanced']
            ))

        return phases

    def _assess_reorganization_risks(self, integrated_intelligence: List) -> Dict[str, Any]:
        """Assess risks associated with the reorganization"""

        risks = []

        # Security risks
        security_modules = len([i for i in integrated_intelligence if 'security' in i.integrated_classification])
        if security_modules > 0:
            risks.append({
                'type': 'Security Risk',
                'severity': 'High',
                'description': f'{security_modules} security-related modules require careful handling',
                'mitigation': 'Conduct security review before any moves'
            })

        # Low confidence risks
        low_confidence = len([i for i in integrated_intelligence if i.integration_confidence < 0.6])
        if low_confidence > 0:
            severity = 'High' if low_confidence > len(integrated_intelligence) * 0.3 else 'Medium'
            risks.append({
                'type': 'Low Confidence Risk',
                'severity': severity,
                'description': f'{low_confidence} modules have low analysis confidence',
                'mitigation': 'Manual review recommended for low-confidence modules'
            })

        # Complexity risks
        complex_modules = len([i for i in integrated_intelligence if i.reorganization_priority >= 8])
        if complex_modules > 0:
            risks.append({
                'type': 'Complexity Risk',
                'severity': 'Medium',
                'description': f'{complex_modules} highly complex modules identified',
                'mitigation': 'Detailed planning and architecture review required'
            })

        return {
            'overall_risk_level': 'High' if any(r['severity'] == 'High' for r in risks) else 'Medium',
            'identified_risks': risks,
            'recommendations': [
                'Start with Phase 1 (low-risk) modules first',
                'Conduct security review for Phase 2 modules',
                'Perform detailed analysis for Phase 3 modules',
                'Maintain comprehensive backups',
                'Test thoroughly after each phase'
            ]
        }

    def calculate_reorganization_priority(self, llm_entry, static_analysis,
                                        confidence_factors) -> int:
        """Calculate reorganization priority for a module"""
        priority = 5  # Base priority

        # Boost for high confidence
        if confidence_factors.agreement_confidence > 0.8:
            priority += 2

        # Boost for security modules
        if 'security' in llm_entry.primary_classification or 'auth' in llm_entry.primary_classification:
            priority += 3

        # Boost for complex modules needing refactoring
        if static_analysis.quality and static_analysis.quality.get('overall_score', 1.0) < 0.6:
            priority += 2

        # Reduce for very low confidence
        if confidence_factors.agreement_confidence < 0.4:
            priority -= 2

        return max(1, min(10, priority))  # Clamp between 1-10

    def generate_integrated_recommendations(self, llm_entry, static_analysis,
                                          confidence_factors, classification_result) -> List[str]:
        """Generate integrated reorganization recommendations"""
        recommendations = []

        primary_class = classification_result.primary_classification
        confidence = classification_result.confidence_score

        if confidence > 0.8:
            recommendations.append(f"High confidence ({confidence:.1%}) - reorganize into '{primary_class}' category")
        elif confidence > 0.6:
            recommendations.append(f"Medium confidence ({confidence:.1%}) - consider reorganization into '{primary_class}' category")
        else:
            recommendations.append(f"Low confidence ({confidence:.1%}) - manual review recommended before reorganization")

        # Add specific recommendations based on analysis
        if static_analysis.quality and static_analysis.quality.get('overall_score', 1.0) < 0.6:
            recommendations.append("Quality issues detected - consider refactoring before or during reorganization")

        if 'security' in primary_class:
            recommendations.append("Security-related module - handle with extra care and review")

        return recommendations
