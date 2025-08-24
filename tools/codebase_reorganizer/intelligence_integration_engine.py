#!/usr/bin/env python3
"""
Intelligence Integration Engine
===============================

Combines LLM analysis with traditional static analysis to create comprehensive
intelligence about Python modules and generate phased reorganization plans.

This engine:
1. Integrates multiple intelligence sources with confidence scoring
2. Generates consensus classifications and reorganization priorities
3. Creates phased reorganization plans with risk assessment
4. Provides actionable insights for codebase reorganization
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

# Import the intelligence system components
try:
    from llm_intelligence_system import (
        LLMIntelligenceScanner, LLMIntelligenceEntry, LLMIntelligenceMap,
        Classification, StaticAnalysisResult, IntegratedIntelligence,
        ReorganizationPhase, ReorganizationPlan
    )
    from semantic_analyzer import SemanticAnalyzer
    from relationship_analyzer import RelationshipAnalyzer
    from pattern_detector import PatternDetector
    from code_quality_analyzer import CodeQualityAnalyzer
    HAS_COMPONENTS = True
except ImportError as e:
    print(f"Warning: Missing components: {e}")
    HAS_COMPONENTS = False


class IntegrationMethod(Enum):
    """Methods for integrating multiple intelligence sources"""
    WEIGHTED_VOTING = "weighted_voting"
    CONSENSUS_WITH_FALLBACK = "consensus_with_fallback"
    LLM_DOMINANT = "llm_dominant"
    STATIC_DOMINANT = "static_dominant"
    ADAPTIVE_CONFIDENCE = "adaptive_confidence"


@dataclass
class ConfidenceFactors:
    """Confidence factors from different analysis sources"""
    llm_confidence: float = 0.0
    semantic_confidence: float = 0.0
    pattern_confidence: float = 0.0
    quality_confidence: float = 0.0
    relationship_confidence: float = 0.0
    agreement_confidence: float = 0.0

    def calculate_overall_confidence(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall confidence using weighted factors"""
        if weights is None:
            weights = {
                'llm_confidence': 0.35,
                'semantic_confidence': 0.20,
                'pattern_confidence': 0.15,
                'quality_confidence': 0.15,
                'relationship_confidence': 0.10,
                'agreement_confidence': 0.05
            }

        weighted_sum = sum(
            getattr(self, factor) * weight
            for factor, weight in weights.items()
            if hasattr(self, factor)
        )

        return min(1.0, max(0.0, weighted_sum))


@dataclass
class ClassificationResult:
    """Result of classification integration"""
    primary_classification: str
    secondary_classifications: List[str]
    confidence_score: float
    reasoning: List[str]
    confidence_factors: ConfidenceFactors
    alternative_classifications: List[Tuple[str, float]]


@dataclass
class ReorganizationRecommendation:
    """Specific reorganization recommendation"""
    source_path: str
    target_path: str
    rationale: str
    confidence: float
    priority: int
    risk_level: str
    dependencies: List[str]
    prerequisites: List[str]
    estimated_effort: int  # minutes


class IntelligenceIntegrationEngine:
    """
    Integrates multiple intelligence sources to provide comprehensive
    analysis and reorganization recommendations.
    """

    def __init__(self, root_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.root_dir = root_dir.resolve()
        self.config = config or self._get_default_config()

        # Setup logging
        self._setup_logging()

        # Initialize analyzers
        self.analyzers = self._initialize_analyzers()

        # Load classification taxonomy
        self.classification_taxonomy = self._load_classification_taxonomy()

        # Initialize integration method
        self.integration_method = IntegrationMethod(self.config.get('integration_method', 'weighted_voting'))

        self.logger.info("Intelligence Integration Engine initialized")
        self.logger.info(f"Integration method: {self.integration_method.value}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'integration_method': 'weighted_voting',
            'confidence_threshold': 0.7,
            'high_confidence_threshold': 0.85,
            'consensus_threshold': 0.6,
            'max_recommendations_per_file': 3,
            'enable_static_analysis': True,
            'classification_weights': {
                'llm_confidence': 0.35,
                'semantic_confidence': 0.20,
                'pattern_confidence': 0.15,
                'quality_confidence': 0.15,
                'relationship_confidence': 0.10,
                'agreement_confidence': 0.05
            },
            'risk_thresholds': {
                'low': 0.8,
                'medium': 0.6,
                'high': 0.4
            }
        }

    def _setup_logging(self) -> None:
        """Setup comprehensive logging"""
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"integration_engine_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_analyzers(self) -> Dict[str, Any]:
        """Initialize static analyzers"""
        analyzers = {}

        if not HAS_COMPONENTS or not self.config['enable_static_analysis']:
            return analyzers

        try:
            analyzers['semantic'] = SemanticAnalyzer()
            analyzers['relationship'] = RelationshipAnalyzer()
            analyzers['pattern'] = PatternDetector()
            analyzers['quality'] = CodeQualityAnalyzer()
            self.logger.info("Static analyzers initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize static analyzers: {e}")

        return analyzers

    def _load_classification_taxonomy(self) -> Dict[str, Any]:
        """Load classification taxonomy and rules"""
        return {
            'primary_categories': [c.value for c in Classification],
            'category_mappings': {
                'auth': 'security',
                'authentication': 'security',
                'authorization': 'security',
                'crypto': 'security',
                'encryption': 'security',
                'security': 'security',
                'test': 'testing',
                'spec': 'testing',
                'mock': 'testing',
                'fixture': 'testing',
                'pytest': 'testing',
                'unittest': 'testing',
                'api': 'api',
                'endpoint': 'api',
                'route': 'api',
                'request': 'api',
                'response': 'api',
                'frontend': 'frontend_dashboard',
                'dashboard': 'frontend_dashboard',
                'ui': 'frontend_dashboard',
                'web': 'frontend_dashboard',
                'interface': 'frontend_dashboard',
                'database': 'database',
                'db': 'database',
                'sql': 'database',
                'model': 'database',
                'orm': 'database',
                'data': 'data_processing',
                'process': 'data_processing',
                'transform': 'data_processing',
                'clean': 'data_processing',
                'extract': 'data_processing',
                'load': 'data_processing',
                'etl': 'data_processing',
                'utility': 'utility',
                'util': 'utility',
                'helper': 'utility',
                'common': 'utility',
                'shared': 'utility',
                'tool': 'utility',
                'automation': 'automation',
                'script': 'automation',
                'job': 'automation',
                'task': 'automation',
                'workflow': 'automation',
                'orchestrate': 'orchestration',
                'coordinator': 'orchestration',
                'manager': 'orchestration',
                'controller': 'orchestration',
                'monitor': 'monitoring',
                'log': 'monitoring',
                'alert': 'monitoring',
                'metric': 'monitoring',
                'telemetry': 'monitoring',
                'observe': 'monitoring',
                'analyze': 'analytics',
                'report': 'analytics',
                'metric': 'analytics',
                'statistic': 'analytics',
                'insight': 'analytics',
                'devops': 'devops',
                'deploy': 'devops',
                'build': 'devops',
                'ci': 'devops',
                'cd': 'devops',
                'infrastructure': 'devops'
            },
            'category_hierarchies': {
                'security': ['authentication', 'authorization', 'encryption', 'validation'],
                'testing': ['unit_tests', 'integration_tests', 'e2e_tests', 'performance_tests'],
                'api': ['rest_api', 'graphql', 'webhook', 'middleware'],
                'frontend_dashboard': ['ui_components', 'dashboard', 'visualization', 'interaction'],
                'database': ['models', 'migrations', 'queries', 'connections'],
                'data_processing': ['etl', 'validation', 'transformation', 'streaming'],
                'utility': ['helpers', 'decorators', 'constants', 'exceptions'],
                'automation': ['scripts', 'jobs', 'workflows', 'scheduling'],
                'orchestration': ['coordination', 'messaging', 'events', 'state_management'],
                'monitoring': ['logging', 'alerting', 'metrics', 'tracing'],
                'analytics': ['reporting', 'insights', 'statistics', 'visualization'],
                'devops': ['deployment', 'infrastructure', 'ci_cd', 'configuration']
            }
        }

    def integrate_intelligence(self, llm_intelligence_map: Dict[str, Any]) -> List[IntegratedIntelligence]:
        """
        Integrate LLM analysis with static analysis to create comprehensive intelligence.

        Args:
            llm_intelligence_map: LLM intelligence map from scanner

        Returns:
            List of integrated intelligence entries
        """
        self.logger.info("Starting intelligence integration...")

        integrated_results = []

        if 'intelligence_entries' not in llm_intelligence_map:
            self.logger.error("Invalid LLM intelligence map format")
            return integrated_results

        entries = llm_intelligence_map['intelligence_entries']
        self.logger.info(f"Processing {len(entries)} intelligence entries")

        for entry_data in entries:
            try:
                # Convert to LLMIntelligenceEntry
                llm_entry = LLMIntelligenceEntry(**entry_data)

                # Perform fresh static analysis
                static_analysis = self._perform_static_analysis(llm_entry.full_path)

                # Integrate the analyses
                integrated = self._integrate_single_entry(llm_entry, static_analysis)

                if integrated:
                    integrated_results.append(integrated)

            except Exception as e:
                self.logger.error(f"Error integrating entry {entry_data.get('full_path', 'unknown')}: {e}")

        self.logger.info(f"Successfully integrated {len(integrated_results)} entries")
        return integrated_results

    def _perform_static_analysis(self, file_path: str) -> StaticAnalysisResult:
        """Perform fresh static analysis on a file"""
        analysis = StaticAnalysisResult()

        if not self.analyzers or not Path(file_path).exists():
            return analysis

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Semantic analysis
            if 'semantic' in self.analyzers:
                analysis.semantic = self.analyzers['semantic'].analyze_semantics(content, Path(file_path))

            # Relationship analysis
            if 'relationship' in self.analyzers:
                analysis.relationship = self.analyzers['relationship'].analyze_relationships(content, file_path)

            # Pattern analysis
            if 'pattern' in self.analyzers:
                analysis.pattern = self.analyzers['pattern'].detect_patterns(content, Path(file_path))

            # Quality analysis
            if 'quality' in self.analyzers:
                analysis.quality = self.analyzers['quality'].analyze_quality(content, file_path)

        except Exception as e:
            self.logger.warning(f"Static analysis failed for {file_path}: {e}")

        return analysis

    def _integrate_single_entry(self, llm_entry: LLMIntelligenceEntry,
                              static_analysis: StaticAnalysisResult) -> Optional[IntegratedIntelligence]:
        """Integrate a single entry's analyses"""

        # Calculate confidence factors
        confidence_factors = self._calculate_confidence_factors(llm_entry, static_analysis)

        # Determine integrated classification
        classification_result = self._determine_integrated_classification(
            llm_entry, static_analysis, confidence_factors
        )

        # Calculate reorganization priority
        reorganization_priority = self._calculate_reorganization_priority(
            llm_entry, static_analysis, classification_result, confidence_factors
        )

        # Generate integrated recommendations
        final_recommendations = self._generate_integrated_recommendations(
            llm_entry, static_analysis, classification_result
        )

        # Calculate overall integration confidence
        integration_confidence = confidence_factors.calculate_overall_confidence(
            self.config['classification_weights']
        )

        # Generate synthesis reasoning
        synthesis_reasoning = self._generate_synthesis_reasoning(
            llm_entry, static_analysis, classification_result, confidence_factors
        )

        return IntegratedIntelligence(
            file_path=llm_entry.full_path,
            relative_path=llm_entry.relative_path,
            static_analysis=static_analysis,
            llm_analysis=llm_entry,
            confidence_factors=confidence_factors,
            integrated_classification=classification_result.primary_classification,
            reorganization_priority=reorganization_priority,
            integration_confidence=integration_confidence,
            final_recommendations=final_recommendations,
            synthesis_reasoning=synthesis_reasoning
        )

    def _calculate_confidence_factors(self, llm_entry: LLMIntelligenceEntry,
                                    static_analysis: StaticAnalysisResult) -> ConfidenceFactors:
        """Calculate confidence factors from different analysis sources"""

        factors = ConfidenceFactors()

        # LLM confidence
        factors.llm_confidence = llm_entry.confidence_score

        # Static analysis confidences
        if static_analysis.semantic:
            factors.semantic_confidence = static_analysis.semantic.get('semantic_confidence', 0.0)

        if static_analysis.pattern:
            pattern_data = static_analysis.pattern
            pattern_confidence = 0
            if 'high_confidence_patterns' in pattern_data:
                pattern_confidence = pattern_data['high_confidence_patterns'] / max(pattern_data.get('total_patterns', 1), 1)
            factors.pattern_confidence = pattern_confidence

        if static_analysis.quality:
            factors.quality_confidence = static_analysis.quality.get('overall_score', 0.5)

        if static_analysis.relationship:
            # Simple heuristic for relationship confidence
            coupling_score = static_analysis.relationship.get('coupling_metrics', {}).get('coupling_score', 0.5)
            factors.relationship_confidence = max(0, 1.0 - coupling_score)  # Lower coupling = higher confidence

        # Calculate agreement confidence
        factors.agreement_confidence = self._calculate_agreement_confidence(llm_entry, static_analysis)

        return factors

    def _calculate_agreement_confidence(self, llm_entry: LLMIntelligenceEntry,
                                      static_analysis: StaticAnalysisResult) -> float:
        """Calculate how well different analyses agree"""

        agreement_score = 0.5  # Base score
        factors = 0

        # Compare LLM classification with semantic purpose
        if static_analysis.semantic:
            semantic_purpose = static_analysis.semantic.get('primary_purpose', '').lower()
            llm_classification = llm_entry.primary_classification.lower()

            # Check for keyword matches
            if any(keyword in semantic_purpose for keyword in llm_classification.split('_')):
                agreement_score += 0.2
            factors += 1

        # Compare complexity assessments
        if static_analysis.quality:
            quality_score = static_analysis.quality.get('overall_score', 0.5)
            llm_complexity = llm_entry.complexity_assessment.lower()

            if 'high' in llm_complexity and quality_score < 0.7:
                agreement_score += 0.2
            elif 'low' in llm_complexity and quality_score > 0.8:
                agreement_score += 0.2
            factors += 1

        # Compare pattern detection with LLM features
        if static_analysis.pattern and llm_entry.key_features:
            pattern_names = [p.get('pattern_name', '').lower() for p in static_analysis.pattern.get('patterns', [])]
            llm_features = [f.lower() for f in llm_entry.key_features]

            matches = 0
            for pattern in pattern_names:
                if any(pattern in feature or feature in pattern for feature in llm_features):
                    matches += 1

            if matches > 0:
                agreement_score += min(0.2, matches * 0.1)
                factors += 1

        return agreement_score / max(factors, 1) if factors > 0 else 0.5

    def _determine_integrated_classification(self, llm_entry: LLMIntelligenceEntry,
                                           static_analysis: StaticAnalysisResult,
                                           confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """Determine the integrated classification using the configured method"""

        if self.integration_method == IntegrationMethod.LLM_DOMINANT:
            return self._llm_dominant_classification(llm_entry, static_analysis, confidence_factors)

        elif self.integration_method == IntegrationMethod.STATIC_DOMINANT:
            return self._static_dominant_classification(llm_entry, static_analysis, confidence_factors)

        elif self.integration_method == IntegrationMethod.CONSENSUS_WITH_FALLBACK:
            return self._consensus_classification(llm_entry, static_analysis, confidence_factors)

        elif self.integration_method == IntegrationMethod.ADAPTIVE_CONFIDENCE:
            return self._adaptive_confidence_classification(llm_entry, static_analysis, confidence_factors)

        else:  # WEIGHTED_VOTING (default)
            return self._weighted_voting_classification(llm_entry, static_analysis, confidence_factors)

    def _weighted_voting_classification(self, llm_entry: LLMIntelligenceEntry,
                                      static_analysis: StaticAnalysisResult,
                                      confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """Use weighted voting to determine classification"""

        # Collect votes from different sources
        votes = defaultdict(float)

        # LLM vote
        votes[llm_entry.primary_classification] += confidence_factors.llm_confidence * 0.35

        # Semantic analysis vote
        if static_analysis.semantic:
            semantic_purpose = static_analysis.semantic.get('primary_purpose', 'unknown')
            mapped_category = self.classification_taxonomy['category_mappings'].get(semantic_purpose, semantic_purpose)
            votes[mapped_category] += confidence_factors.semantic_confidence * 0.20

        # Pattern analysis vote
        if static_analysis.pattern:
            pattern_votes = self._get_pattern_based_classification(static_analysis.pattern)
            for category, confidence in pattern_votes.items():
                votes[category] += confidence * 0.15

        # Quality analysis vote (lower weight)
        if static_analysis.quality:
            quality_category = self._get_quality_based_classification(static_analysis.quality)
            votes[quality_category] += confidence_factors.quality_confidence * 0.10

        # Determine winner
        if votes:
            winner = max(votes.items(), key=lambda x: x[1])
            primary_classification = winner[0]
            confidence = min(1.0, winner[1])

            # Get secondary classifications
            threshold = winner[1] * 0.6
            secondary_classifications = [cat for cat, score in votes.items()
                                       if score >= threshold and cat != primary_classification]

            # Alternative classifications
            alternative_classifications = sorted(votes.items(), key=lambda x: x[1], reverse=True)[1:4]
        else:
            primary_classification = 'uncategorized'
            confidence = 0.5
            secondary_classifications = []
            alternative_classifications = []

        return ClassificationResult(
            primary_classification=primary_classification,
            secondary_classifications=secondary_classifications,
            confidence_score=confidence,
            reasoning=self._generate_classification_reasoning(votes, confidence_factors),
            confidence_factors=confidence_factors,
            alternative_classifications=alternative_classifications
        )

    def _llm_dominant_classification(self, llm_entry: LLMIntelligenceEntry,
                                   static_analysis: StaticAnalysisResult,
                                   confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """LLM-dominant classification with static analysis as validation"""

        primary_classification = llm_entry.primary_classification
        confidence = llm_entry.confidence_score

        # Boost confidence if static analysis agrees
        if confidence_factors.agreement_confidence > 0.7:
            confidence = min(1.0, confidence + 0.1)

        return ClassificationResult(
            primary_classification=primary_classification,
            secondary_classifications=llm_entry.secondary_classifications,
            confidence_score=confidence,
            reasoning=["LLM-dominant classification with static validation"],
            confidence_factors=confidence_factors,
            alternative_classifications=[]
        )

    def _consensus_classification(self, llm_entry: LLMIntelligenceEntry,
                                static_analysis: StaticAnalysisResult,
                                confidence_factors: ConfidenceFactors) -> ClassificationResult:
        """Consensus-based classification requiring agreement"""

        # Use weighted voting but require minimum consensus
        weighted_result = self._weighted_voting_classification(llm_entry, static_analysis, confidence_factors)

        if confidence_factors.agreement_confidence < self.config['consensus_threshold']:
            # Low agreement - use LLM but reduce confidence
            return ClassificationResult(
                primary_classification=llm_entry.primary_classification,
                secondary_classifications=llm_entry.secondary_classifications,
                confidence_score=llm_entry.confidence_score * 0.7,
                reasoning=["Low consensus - using LLM with reduced confidence"],
                confidence_factors=confidence_factors,
                alternative_classifications=weighted_result.alternative_classifications
            )

        return weighted_result

    def _get_pattern_based_classification(self, pattern_data: Dict[str, Any]) -> Dict[str, float]:
        """Get classification suggestions from pattern analysis"""
        votes = defaultdict(float)

        patterns = pattern_data.get('patterns', [])
        for pattern in patterns:
            pattern_name = pattern.get('pattern_name', '').lower()

            # Map patterns to categories
            if 'singleton' in pattern_name:
                votes['utility'] += 0.8
            elif 'factory' in pattern_name:
                votes['utility'] += 0.7
            elif 'observer' in pattern_name:
                votes['orchestration'] += 0.6
            elif 'strategy' in pattern_name:
                votes['utility'] += 0.6
            elif 'decorator' in pattern_name:
                votes['utility'] += 0.7

        return dict(votes)

    def _get_quality_based_classification(self, quality_data: Dict[str, Any]) -> str:
        """Get classification suggestion from quality analysis"""
        score = quality_data.get('overall_score', 0.5)

        if score < 0.6:
            return 'needs_refactoring'
        else:
            return 'utility'  # Default assumption

    def _generate_classification_reasoning(self, votes: Dict[str, float],
                                         confidence_factors: ConfidenceFactors) -> List[str]:
        """Generate reasoning for classification decision"""
        reasoning = []

        # Add top vote explanation
        if votes:
            winner = max(votes.items(), key=lambda x: x[1])
            reasoning.append(f"Primary classification '{winner[0]}' with weighted score {winner[1]:.3f}")

        # Add confidence factor explanations
        if confidence_factors.llm_confidence > 0.8:
            reasoning.append(f"Strong LLM confidence ({confidence_factors.llm_confidence:.2f})")
        elif confidence_factors.llm_confidence < 0.5:
            reasoning.append(f"Weak LLM confidence ({confidence_factors.llm_confidence:.2f})")

        if confidence_factors.agreement_confidence > 0.7:
            reasoning.append("High agreement between analysis methods")
        elif confidence_factors.agreement_confidence < 0.4:
            reasoning.append("Low agreement between analysis methods")

        return reasoning

    def _calculate_reorganization_priority(self, llm_entry: LLMIntelligenceEntry,
                                         static_analysis: StaticAnalysisResult,
                                         classification_result: ClassificationResult,
                                         confidence_factors: ConfidenceFactors) -> int:
        """Calculate reorganization priority (1-10)"""

        priority = 5  # Base priority

        # LLM-based factors
        if llm_entry.confidence_score > 0.8:
            priority += 2
        elif llm_entry.confidence_score < 0.5:
            priority -= 1

        # Security factors
        if (llm_entry.security_implications and
            'none' not in llm_entry.security_implications.lower()):
            priority += 2

        if 'security' in classification_result.primary_classification:
            priority += 1

        # Quality factors
        if static_analysis.quality:
            quality_score = static_analysis.quality.get('overall_score', 0.5)
            if quality_score < 0.6:
                priority += 2  # Poor quality needs attention
            elif quality_score > 0.8:
                priority -= 1  # Good quality can wait

        # Integration confidence
        integration_confidence = confidence_factors.calculate_overall_confidence(
            self.config['classification_weights']
        )

        if integration_confidence > 0.8:
            priority += 1
        elif integration_confidence < 0.6:
            priority -= 1

        # Size factors
        if llm_entry.complexity_assessment.lower() in ['high', 'very_high']:
            priority += 1

        # Consensus factors
        if confidence_factors.agreement_confidence > 0.7:
            priority += 1
        elif confidence_factors.agreement_confidence < 0.4:
            priority -= 1

        return max(1, min(10, priority))

    def _generate_integrated_recommendations(self, llm_entry: LLMIntelligenceEntry,
                                           static_analysis: StaticAnalysisResult,
                                           classification_result: ClassificationResult) -> List[str]:
        """Generate integrated reorganization recommendations"""

        recommendations = []
        recommendations.extend(llm_entry.reorganization_recommendations)

        # Add quality-based recommendations
        if static_analysis.quality:
            quality_data = static_analysis.quality
            if quality_data.get('overall_score', 1.0) < 0.7:
                recommendations.append("Address code quality issues before reorganization")
                quality_issues = quality_data.get('critical_issues', [])
                recommendations.extend([f"Fix: {issue}" for issue in quality_issues[:2]])

        # Add pattern-based recommendations
        if static_analysis.pattern:
            patterns = static_analysis.pattern.get('patterns', [])
            if patterns:
                recommendations.append("Consider applying detected patterns consistently")

        # Add relationship-based recommendations
        if static_analysis.relationship:
            coupling_score = static_analysis.relationship.get('coupling_metrics', {}).get('coupling_score', 0.5)
            if coupling_score > 0.7:
                recommendations.append("High coupling detected - consider breaking dependencies")

        # Add classification-based recommendations
        if classification_result.primary_classification != llm_entry.primary_classification:
            recommendations.append(f"Classification updated from '{llm_entry.primary_classification}' to '{classification_result.primary_classification}' based on integrated analysis")

        # Add confidence-based recommendations
        if classification_result.confidence_score < 0.6:
            recommendations.append("Low confidence in classification - manual review recommended")

        return recommendations[:self.config['max_recommendations_per_file']]

    def _generate_synthesis_reasoning(self, llm_entry: LLMIntelligenceEntry,
                                    static_analysis: StaticAnalysisResult,
                                    classification_result: ClassificationResult,
                                    confidence_factors: ConfidenceFactors) -> str:
        """Generate reasoning for the synthesis"""

        reasoning_parts = []

        # Classification reasoning
        reasoning_parts.append(f"Classification: {classification_result.primary_classification} (confidence: {classification_result.confidence_score:.2f})")

        # Confidence factors
        high_factors = []
        if confidence_factors.llm_confidence > 0.8:
            high_factors.append("LLM")
        if confidence_factors.semantic_confidence > 0.7:
            high_factors.append("Semantic")
        if confidence_factors.pattern_confidence > 0.6:
            high_factors.append("Pattern")
        if confidence_factors.quality_confidence > 0.7:
            high_factors.append("Quality")

        if high_factors:
            reasoning_parts.append(f"High confidence from: {', '.join(high_factors)}")

        # Agreement level
        if confidence_factors.agreement_confidence > 0.7:
            reasoning_parts.append("Strong agreement between analysis methods")
        elif confidence_factors.agreement_confidence < 0.4:
            reasoning_parts.append("Disagreement between analysis methods")

        return " | ".join(reasoning_parts)

    def generate_reorganization_plan(self, integrated_intelligence: List[IntegratedIntelligence],
                                   llm_intelligence_map: Dict[str, Any]) -> ReorganizationPlan:
        """Generate a comprehensive reorganization plan"""

        self.logger.info("Generating reorganization plan...")

        # Group by classification
        by_classification = defaultdict(list)
        for intelligence in integrated_intelligence:
            by_classification[intelligence.integrated_classification].append(intelligence)

        # Calculate statistics
        total_modules = len(integrated_intelligence)
        high_priority = len([i for i in integrated_intelligence if i.reorganization_priority >= 8])
        security_modules = len([i for i in integrated_intelligence if 'security' in i.integrated_classification])

        # Generate phases
        reorganization_phases = self._generate_reorganization_phases(integrated_intelligence, by_classification)

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

        plan = ReorganizationPlan(
            plan_timestamp=datetime.now().isoformat(),
            total_modules=total_modules,
            reorganization_phases=reorganization_phases,
            estimated_total_time_hours=estimated_hours,
            risk_assessment=risk_assessment,
            success_metrics=success_metrics,
            implementation_guidelines=implementation_guidelines
        )

        self.logger.info(f"Generated reorganization plan with {len(reorganization_phases)} phases")
        return plan

    def _generate_reorganization_phases(self, integrated_intelligence: List[IntegratedIntelligence],
                                      by_classification: Dict[str, List[IntegratedIntelligence]]) -> List[ReorganizationPhase]:
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
            phases.append(ReorganizationPhase(
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
            phases.append(ReorganizationPhase(
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
            phases.append(ReorganizationPhase(
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

    def _assess_reorganization_risks(self, integrated_intelligence: List[IntegratedIntelligence]) -> Dict[str, Any]:
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
                'type': 'Analysis Confidence Risk',
                'severity': severity,
                'description': f'{low_confidence} modules have low analysis confidence',
                'mitigation': 'Manual review required for low-confidence modules'
            })

        # Complexity risks
        complex_modules = len([i for i in integrated_intelligence
                             if 'high' in i.llm_analysis.complexity_assessment.lower()])
        if complex_modules > 0:
            risks.append({
                'type': 'Complexity Risk',
                'severity': 'Medium',
                'description': f'{complex_modules} complex modules may require additional testing',
                'mitigation': 'Ensure comprehensive testing after reorganization'
            })

        # Overall risk assessment
        overall_risk = 'Low'
        if any(r['severity'] == 'High' for r in risks):
            overall_risk = 'High'
        elif any(r['severity'] == 'Medium' for r in risks):
            overall_risk = 'Medium'

        return {
            'risks': risks,
            'overall_risk_level': overall_risk,
            'recommendations': [r['mitigation'] for r in risks],
            'risk_summary': f"Identified {len(risks)} risk categories with overall risk level: {overall_risk}"
        }

    def save_integration_results(self, integrated_intelligence: List[IntegratedIntelligence],
                               reorganization_plan: ReorganizationPlan,
                               output_file: Path) -> None:
        """Save integration results to file"""

        results = {
            'integration_timestamp': datetime.now().isoformat(),
            'integrated_intelligence': [asdict(intel) for intel in integrated_intelligence],
            'reorganization_plan': asdict(reorganization_plan),
            'summary': {
                'total_modules': len(integrated_intelligence),
                'high_priority_modules': len([i for i in integrated_intelligence if i.reorganization_priority >= 8]),
                'security_modules': len([i for i in integrated_intelligence if 'security' in i.integrated_classification]),
                'low_confidence_modules': len([i for i in integrated_intelligence if i.integration_confidence < 0.6]),
                'phases': len(reorganization_plan.reorganization_phases),
                'estimated_hours': reorganization_plan.estimated_total_time_hours
            }
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Integration results saved to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save integration results: {e}")


def main():
    """Main function to run the integration engine"""
    import argparse

    parser = argparse.ArgumentParser(description="Intelligence Integration Engine")
    parser.add_argument("--root", type=str, default=".",
                      help="Root directory")
    parser.add_argument("--llm-map", type=str, required=True,
                      help="Path to LLM intelligence map JSON file")
    parser.add_argument("--output", type=str, default="integrated_intelligence.json",
                      help="Output file for integrated intelligence")
    parser.add_argument("--method", type=str, default="weighted_voting",
                      choices=["weighted_voting", "consensus_with_fallback", "llm_dominant", "static_dominant", "adaptive_confidence"],
                      help="Integration method to use")
    parser.add_argument("--no-static", action="store_true",
                      help="Disable static analysis")

    args = parser.parse_args()

    # Load LLM intelligence map
    try:
        with open(args.llm_map, 'r', encoding='utf-8') as f:
            llm_map = json.load(f)
    except Exception as e:
        print(f"Error loading LLM map: {e}")
        return

    # Create configuration
    config = {
        'integration_method': args.method,
        'enable_static_analysis': not args.no_static,
        'confidence_threshold': 0.7,
        'high_confidence_threshold': 0.85,
        'consensus_threshold': 0.6,
        'max_recommendations_per_file': 3,
        'classification_weights': {
            'llm_confidence': 0.35,
            'semantic_confidence': 0.20,
            'pattern_confidence': 0.15,
            'quality_confidence': 0.15,
            'relationship_confidence': 0.10,
            'agreement_confidence': 0.05
        },
        'risk_thresholds': {
            'low': 0.8,
            'medium': 0.6,
            'high': 0.4
        }
    }

    # Initialize integration engine
    root_dir = Path(args.root).resolve()
    engine = IntelligenceIntegrationEngine(root_dir, config)

    print("🔗 Integrating LLM and Traditional Intelligence...")
    print(f"Integration method: {args.method}")
    print(f"Static analysis: {'enabled' if not args.no_static else 'disabled'}")

    # Perform integration
    integrated_intelligence = engine.integrate_intelligence(llm_map)

    # Generate reorganization plan
    reorganization_plan = engine.generate_reorganization_plan(integrated_intelligence, llm_map)

    # Save results
    engine.save_integration_results(integrated_intelligence, reorganization_plan, Path(args.output))

    print("\n✅ Integration completed!")
    print(f"Integrated {len(integrated_intelligence)} modules")
    print(f"Generated reorganization plan with {len(reorganization_plan.reorganization_phases)} phases")
    print(".1f")
    print(f"Output saved to: {args.output}")

    # Print summary
    if integrated_intelligence:
        print("\n📊 Integration Summary:")
        high_priority = len([i for i in integrated_intelligence if i.reorganization_priority >= 8])
        security_modules = len([i for i in integrated_intelligence if 'security' in i.integrated_classification])
        low_confidence = len([i for i in integrated_intelligence if i.integration_confidence < 0.6])

        print(f"  High priority modules: {high_priority}")
        print(f"  Security modules: {security_modules}")
        print(f"  Low confidence modules: {low_confidence}")

        # Classification distribution
        from collections import Counter
        classifications = Counter(i.integrated_classification for i in integrated_intelligence)
        print("  Top classifications:")
        for classification, count in classifications.most_common(5):
            print(f"    {classification}: {count}")


if __name__ == "__main__":
    main()

