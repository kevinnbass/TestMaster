"""
Technology Trend Analyzer
=========================

Analyzes technology trends and their impact on architecture evolution.
Extracted from architectural_evolution_predictor.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import logging
from typing import Dict, List, Any

from .data_models import ArchitecturalMetrics, TechnologyEvolutionAnalysis, TechnologyTrend

# Configure logging
logger = logging.getLogger(__name__)


class TechnologyTrendAnalyzer:
    """Analyzes technology trends and their impact on architecture"""
    
    def __init__(self):
        self.technology_trends = {
            TechnologyTrend.CLOUD_NATIVE: {
                'adoption_rate': 0.7,
                'maturity': 0.8,
                'impact_areas': ['scalability', 'deployment', 'operations'],
                'investment_level': 'medium'
            },
            TechnologyTrend.AI_ML_INTEGRATION: {
                'adoption_rate': 0.5,
                'maturity': 0.6,
                'impact_areas': ['intelligence', 'automation', 'decision_making'],
                'investment_level': 'high'
            },
            TechnologyTrend.EDGE_COMPUTING: {
                'adoption_rate': 0.3,
                'maturity': 0.5,
                'impact_areas': ['performance', 'latency', 'distribution'],
                'investment_level': 'medium'
            },
            TechnologyTrend.SERVERLESS_COMPUTING: {
                'adoption_rate': 0.4,
                'maturity': 0.6,
                'impact_areas': ['cost', 'scalability', 'operations'],
                'investment_level': 'low'
            },
            TechnologyTrend.CONTAINERIZATION: {
                'adoption_rate': 0.8,
                'maturity': 0.9,
                'impact_areas': ['deployment', 'scalability', 'portability'],
                'investment_level': 'medium'
            }
        }
        
        self.impact_weights = {
            'performance': 0.25,
            'scalability': 0.20,
            'cost': 0.15,
            'development_velocity': 0.15,
            'operational_efficiency': 0.10,
            'security': 0.10,
            'innovation': 0.05
        }
    
    def analyze_technology_trends(self, current_architecture: ArchitecturalMetrics, 
                                system_requirements: Dict[str, Any]) -> TechnologyEvolutionAnalysis:
        """Analyze technology trends and their relevance to the system"""
        try:
            analysis = TechnologyEvolutionAnalysis()
            
            # Extract current technology stack (simplified)
            analysis.current_technology_stack = self._extract_technology_stack(current_architecture)
            
            # Identify relevant emerging trends
            analysis.emerging_trends = list(self.technology_trends.keys())
            
            # Calculate relevance scores for each trend
            analysis.trend_relevance_scores = self._calculate_trend_relevance(
                current_architecture, system_requirements
            )
            
            # Determine adoption timelines
            analysis.adoption_timeline = self._determine_adoption_timeline(
                analysis.trend_relevance_scores, current_architecture
            )
            
            # Assess migration complexity
            analysis.migration_complexity = self._assess_migration_complexity(
                analysis.current_technology_stack, analysis.emerging_trends
            )
            
            # Analyze business impact
            analysis.business_impact = self._analyze_business_impact(
                analysis.emerging_trends, system_requirements
            )
            
            # Assess technical feasibility
            analysis.technical_feasibility = self._assess_technical_feasibility(
                analysis.emerging_trends, current_architecture
            )
            
            # Perform risk assessment
            analysis.risk_assessment = self._perform_risk_assessment(
                analysis.emerging_trends, current_architecture
            )
            
            # Calculate investment requirements
            analysis.investment_requirements = self._calculate_investment_requirements(
                analysis.emerging_trends, analysis.migration_complexity
            )
            
            # Assess competitive advantage
            analysis.competitive_advantage = self._assess_competitive_advantage(
                analysis.emerging_trends, analysis.trend_relevance_scores
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing technology trends: {e}")
            return TechnologyEvolutionAnalysis()
    
    def _extract_technology_stack(self, architecture: ArchitecturalMetrics) -> Dict[str, str]:
        """Extract current technology stack information"""
        try:
            # Simplified technology stack extraction
            stack = {
                'architecture_pattern': 'monolithic' if architecture.service_count < 3 else 'microservices',
                'deployment': 'traditional' if architecture.deployment_complexity < 0.5 else 'cloud',
                'data_storage': 'relational' if architecture.database_count < 3 else 'multi_model',
                'api_style': 'rest',  # Assume REST by default
                'monitoring': 'basic' if architecture.monitoring_coverage < 0.7 else 'advanced'
            }
            return stack
        except Exception as e:
            logger.error(f"Error extracting technology stack: {e}")
            return {}
    
    def _calculate_trend_relevance(self, architecture: ArchitecturalMetrics, 
                                 requirements: Dict[str, Any]) -> Dict[TechnologyTrend, float]:
        """Calculate relevance score for each technology trend"""
        try:
            relevance_scores = {}
            
            for trend in self.technology_trends:
                score = 0.0
                
                if trend == TechnologyTrend.CLOUD_NATIVE:
                    # High relevance for systems needing scalability
                    if architecture.scalability_score < 0.7:
                        score += 0.4
                    if requirements.get('scalability_priority', 'medium') == 'high':
                        score += 0.3
                    if architecture.deployment_complexity > 0.6:
                        score += 0.3
                
                elif trend == TechnologyTrend.AI_ML_INTEGRATION:
                    # High relevance for data-intensive systems
                    if architecture.lines_of_code > 50000:
                        score += 0.3
                    if requirements.get('intelligence_requirements', False):
                        score += 0.5
                    if architecture.component_count > 10:
                        score += 0.2
                
                elif trend == TechnologyTrend.EDGE_COMPUTING:
                    # High relevance for latency-sensitive systems
                    if requirements.get('latency_requirements', 'medium') == 'low':
                        score += 0.6
                    if requirements.get('geographic_distribution', False):
                        score += 0.4
                
                elif trend == TechnologyTrend.SERVERLESS_COMPUTING:
                    # High relevance for cost-sensitive systems
                    if requirements.get('cost_optimization', 'medium') == 'high':
                        score += 0.4
                    if architecture.service_count < 5:
                        score += 0.3
                    if requirements.get('operational_simplicity', False):
                        score += 0.3
                
                else:
                    # Default scoring for other trends
                    score = 0.5
                
                relevance_scores[trend] = min(1.0, score)
            
            return relevance_scores
            
        except Exception as e:
            logger.error(f"Error calculating trend relevance: {e}")
            return {}
    
    def _determine_adoption_timeline(self, relevance_scores: Dict[TechnologyTrend, float], 
                                   architecture: ArchitecturalMetrics) -> Dict[TechnologyTrend, str]:
        """Determine adoption timeline for each trend"""
        try:
            timelines = {}
            
            for trend, relevance in relevance_scores.items():
                trend_info = self.technology_trends.get(trend, {})
                maturity = trend_info.get('maturity', 0.5)
                
                # Calculate timeline based on relevance and maturity
                if relevance > 0.8 and maturity > 0.7:
                    timeline = "3-6 months"
                elif relevance > 0.6 and maturity > 0.6:
                    timeline = "6-12 months"
                elif relevance > 0.4:
                    timeline = "1-2 years"
                else:
                    timeline = "2+ years"
                
                # Adjust based on system complexity
                if architecture.complexity_score > 0.8:
                    # Complex systems take longer to adopt new technologies
                    timeline_map = {
                        "3-6 months": "6-12 months",
                        "6-12 months": "1-2 years",
                        "1-2 years": "2+ years",
                        "2+ years": "3+ years"
                    }
                    timeline = timeline_map.get(timeline, timeline)
                
                timelines[trend] = timeline
            
            return timelines
            
        except Exception as e:
            logger.error(f"Error determining adoption timeline: {e}")
            return {}
    
    def _assess_migration_complexity(self, current_stack: Dict[str, str], 
                                   trends: List[TechnologyTrend]) -> Dict[TechnologyTrend, float]:
        """Assess complexity of migrating to each technology trend"""
        try:
            complexity_scores = {}
            
            for trend in trends:
                complexity = 0.5  # Base complexity
                
                if trend == TechnologyTrend.CLOUD_NATIVE:
                    if current_stack.get('deployment') == 'traditional':
                        complexity = 0.7  # Significant migration needed
                    else:
                        complexity = 0.3  # Already cloud-ready
                
                elif trend == TechnologyTrend.AI_ML_INTEGRATION:
                    # AI/ML integration complexity depends on data architecture
                    if current_stack.get('data_storage') == 'multi_model':
                        complexity = 0.4  # Good data foundation
                    else:
                        complexity = 0.8  # Need data architecture changes
                
                elif trend == TechnologyTrend.EDGE_COMPUTING:
                    # Edge computing requires distributed architecture
                    if current_stack.get('architecture_pattern') == 'microservices':
                        complexity = 0.6  # Moderate complexity
                    else:
                        complexity = 0.9  # High complexity for monoliths
                
                elif trend == TechnologyTrend.SERVERLESS_COMPUTING:
                    # Serverless migration complexity
                    if current_stack.get('architecture_pattern') == 'microservices':
                        complexity = 0.4  # Easier migration
                    else:
                        complexity = 0.8  # Requires decomposition
                
                complexity_scores[trend] = complexity
            
            return complexity_scores
            
        except Exception as e:
            logger.error(f"Error assessing migration complexity: {e}")
            return {}
    
    def _analyze_business_impact(self, trends: List[TechnologyTrend], 
                               requirements: Dict[str, Any]) -> Dict[TechnologyTrend, Dict[str, str]]:
        """Analyze business impact of adopting each technology trend"""
        try:
            business_impacts = {}
            
            for trend in trends:
                impact = {}
                
                if trend == TechnologyTrend.CLOUD_NATIVE:
                    impact = {
                        'cost': 'potentially_reduced',
                        'time_to_market': 'improved',
                        'scalability': 'significantly_improved',
                        'reliability': 'improved',
                        'innovation_speed': 'improved'
                    }
                
                elif trend == TechnologyTrend.AI_ML_INTEGRATION:
                    impact = {
                        'competitive_advantage': 'significant',
                        'automation': 'high',
                        'decision_making': 'enhanced',
                        'operational_efficiency': 'improved',
                        'innovation_potential': 'high'
                    }
                
                elif trend == TechnologyTrend.EDGE_COMPUTING:
                    impact = {
                        'user_experience': 'improved',
                        'latency': 'reduced',
                        'bandwidth_costs': 'reduced',
                        'data_privacy': 'enhanced',
                        'offline_capability': 'enabled'
                    }
                
                elif trend == TechnologyTrend.SERVERLESS_COMPUTING:
                    impact = {
                        'operational_costs': 'reduced',
                        'development_velocity': 'improved',
                        'scalability': 'automated',
                        'operational_overhead': 'reduced',
                        'vendor_lock_in': 'increased'
                    }
                
                else:
                    impact = {
                        'overall': 'moderate_improvement'
                    }
                
                business_impacts[trend] = impact
            
            return business_impacts
            
        except Exception as e:
            logger.error(f"Error analyzing business impact: {e}")
            return {}
    
    def _assess_technical_feasibility(self, trends: List[TechnologyTrend], 
                                    architecture: ArchitecturalMetrics) -> Dict[TechnologyTrend, float]:
        """Assess technical feasibility of adopting each trend"""
        try:
            feasibility_scores = {}
            
            for trend in trends:
                score = 0.5  # Base feasibility
                
                # Adjust based on current architecture maturity
                if architecture.maintainability_score > 0.7:
                    score += 0.2  # Well-maintained systems are easier to evolve
                
                if architecture.test_coverage > 0.8:
                    score += 0.1  # Good test coverage reduces risk
                
                if architecture.monitoring_coverage > 0.7:
                    score += 0.1  # Good monitoring helps with adoption
                
                # Trend-specific adjustments
                if trend == TechnologyTrend.CLOUD_NATIVE:
                    if architecture.deployment_complexity < 0.5:
                        score += 0.2  # Simple deployment makes cloud adoption easier
                
                elif trend == TechnologyTrend.AI_ML_INTEGRATION:
                    if architecture.component_count > 5:
                        score += 0.1  # More components provide more integration points
                
                # Cap the score at 1.0
                feasibility_scores[trend] = min(1.0, score)
            
            return feasibility_scores
            
        except Exception as e:
            logger.error(f"Error assessing technical feasibility: {e}")
            return {}
    
    def _perform_risk_assessment(self, trends: List[TechnologyTrend], 
                               architecture: ArchitecturalMetrics) -> Dict[TechnologyTrend, Dict[str, float]]:
        """Perform risk assessment for adopting each technology trend"""
        try:
            risk_assessments = {}
            
            for trend in trends:
                risks = {
                    'technical_risk': 0.3,      # Base technical risk
                    'business_risk': 0.2,       # Base business risk
                    'operational_risk': 0.2,    # Base operational risk
                    'financial_risk': 0.2,      # Base financial risk
                    'security_risk': 0.1        # Base security risk
                }
                
                # Adjust risks based on system characteristics
                if architecture.complexity_score > 0.8:
                    risks['technical_risk'] += 0.3  # Complex systems have higher technical risk
                
                if architecture.security_score < 0.6:
                    risks['security_risk'] += 0.3  # Poor security increases risk
                
                # Trend-specific risk adjustments
                if trend == TechnologyTrend.AI_ML_INTEGRATION:
                    risks['technical_risk'] += 0.2  # AI/ML has higher technical complexity
                    risks['security_risk'] += 0.1  # AI/ML may introduce security concerns
                
                elif trend == TechnologyTrend.EDGE_COMPUTING:
                    risks['operational_risk'] += 0.3  # Edge computing increases operational complexity
                    risks['security_risk'] += 0.2  # Distributed systems have more attack surface
                
                elif trend == TechnologyTrend.SERVERLESS_COMPUTING:
                    risks['business_risk'] += 0.2  # Vendor lock-in concerns
                    risks['operational_risk'] -= 0.1  # Reduced operational overhead
                
                # Ensure risks don't exceed 1.0
                for risk_type in risks:
                    risks[risk_type] = min(1.0, max(0.0, risks[risk_type]))
                
                risk_assessments[trend] = risks
            
            return risk_assessments
            
        except Exception as e:
            logger.error(f"Error performing risk assessment: {e}")
            return {}
    
    def _calculate_investment_requirements(self, trends: List[TechnologyTrend], 
                                         migration_complexity: Dict[TechnologyTrend, float]) -> Dict[TechnologyTrend, Dict[str, float]]:
        """Calculate investment requirements for each technology trend"""
        try:
            investment_requirements = {}
            
            for trend in trends:
                complexity = migration_complexity.get(trend, 0.5)
                trend_info = self.technology_trends.get(trend, {})
                base_investment = {'low': 0.2, 'medium': 0.5, 'high': 0.8}.get(
                    trend_info.get('investment_level', 'medium'), 0.5
                )
                
                # Calculate investment in different areas
                requirements = {
                    'development_effort': base_investment * complexity,
                    'infrastructure_cost': base_investment * 0.7,
                    'training_cost': base_investment * 0.3,
                    'consulting_cost': complexity * 0.5,
                    'total_investment': 0.0
                }
                
                requirements['total_investment'] = sum(requirements.values()) - requirements['total_investment']
                
                investment_requirements[trend] = requirements
            
            return investment_requirements
            
        except Exception as e:
            logger.error(f"Error calculating investment requirements: {e}")
            return {}
    
    def _assess_competitive_advantage(self, trends: List[TechnologyTrend], 
                                    relevance_scores: Dict[TechnologyTrend, float]) -> Dict[TechnologyTrend, float]:
        """Assess competitive advantage of adopting each technology trend"""
        try:
            advantage_scores = {}
            
            for trend in trends:
                trend_info = self.technology_trends.get(trend, {})
                adoption_rate = trend_info.get('adoption_rate', 0.5)
                relevance = relevance_scores.get(trend, 0.5)
                
                # Early adoption of relevant, low-adoption technologies provides more advantage
                if adoption_rate < 0.3 and relevance > 0.7:
                    advantage = 0.9  # High advantage for early adoption of relevant technology
                elif adoption_rate < 0.5 and relevance > 0.5:
                    advantage = 0.7  # Good advantage
                elif adoption_rate > 0.8:
                    advantage = 0.2  # Low advantage for widely adopted technologies
                else:
                    advantage = 0.5  # Moderate advantage
                
                advantage_scores[trend] = advantage
            
            return advantage_scores
            
        except Exception as e:
            logger.error(f"Error assessing competitive advantage: {e}")
            return {}


__all__ = ['TechnologyTrendAnalyzer']