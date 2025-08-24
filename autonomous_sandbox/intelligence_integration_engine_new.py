#!/usr/bin/env python3
"""
Intelligence Integration Engine - Modular Coordinator
=====================================================

Main coordinator for the comprehensive LLM-based code intelligence system.
This is the modular version that uses focused components.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import our modular components
from integration_models import (
    IntegrationMethod, ConfidenceFactors, ClassificationResult,
    ReorganizationRecommendation
)
from integration_classification import IntelligenceClassificationEngine
from integration_planning import IntelligencePlanningEngine

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

        # Initialize our modular components
        self.classification_engine = IntelligenceClassificationEngine(self.config, self.classification_taxonomy)
        self.planning_engine = IntelligencePlanningEngine(self.config)

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
                'high_risk': 0.8,
                'medium_risk': 0.6,
                'low_risk': 0.4
            }
        }

    def _setup_logging(self) -> None:
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"integration_engine_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _initialize_analyzers(self) -> Dict[str, Any]:
        """Initialize static analysis components"""
        analyzers = {}

        if HAS_COMPONENTS:
            try:
                analyzers['semantic'] = SemanticAnalyzer(self.root_dir)
            except:
                self.logger.warning("Could not initialize semantic analyzer")

            try:
                analyzers['relationship'] = RelationshipAnalyzer(self.root_dir)
            except:
                self.logger.warning("Could not initialize relationship analyzer")

            try:
                analyzers['pattern'] = PatternDetector(self.root_dir)
            except:
                self.logger.warning("Could not initialize pattern detector")

            try:
                analyzers['quality'] = CodeQualityAnalyzer(self.root_dir)
            except:
                self.logger.warning("Could not initialize quality analyzer")

        return analyzers

    def _load_classification_taxonomy(self) -> Dict[str, Any]:
        """Load classification taxonomy"""
        return {
            'category_mappings': {
                'data_processing': 'data',
                'api_client': 'api',
                'configuration': 'config',
                'testing': 'test',
                'documentation': 'docs',
                'security': 'security',
                'orchestration': 'orchestration',
                'intelligence': 'intelligence',
                'utility': 'utilities',
                'monitoring': 'monitoring'
            },
            'priority_mappings': {
                'security': 10,
                'api': 8,
                'orchestration': 7,
                'intelligence': 6,
                'data': 5,
                'config': 4,
                'utilities': 3,
                'docs': 2,
                'test': 1
            }
        }

    def integrate_intelligence(self, llm_intelligence_map: Dict[str, Any]) -> List[IntegratedIntelligence]:
        """Integrate LLM intelligence with static analysis results"""
        self.logger.info("Starting intelligence integration...")

        integrated_results = []

        # Convert map data to entries if needed
        if hasattr(llm_intelligence_map, '__dict__'):
            entries = llm_intelligence_map.entries if hasattr(llm_intelligence_map, 'entries') else []
        else:
            entries = llm_intelligence_map.get('entries', [])

        self.logger.info(f"Processing {len(entries)} intelligence entries")

        for i, entry_data in enumerate(entries):
            try:
                # Convert to LLMIntelligenceEntry if needed
                if isinstance(entry_data, dict):
                    entry = self._dict_to_entry(entry_data)
                else:
                    entry = entry_data

                # Perform static analysis
                static_analysis = self._perform_static_analysis(entry.file_path)

                # Calculate confidence factors
                confidence_factors = self._calculate_confidence_factors(entry, static_analysis)

                # Integrate classifications
                classification_result = self.classification_engine.determine_integrated_classification(
                    self.integration_method, entry, static_analysis, confidence_factors
                )

                # Calculate reorganization priority
                reorganization_priority = self.planning_engine.calculate_reorganization_priority(
                    entry, static_analysis, confidence_factors
                )

                # Generate integrated recommendations
                final_recommendations = self.planning_engine.generate_integrated_recommendations(
                    entry, static_analysis, confidence_factors, classification_result
                )

                # Create integrated intelligence result
                integrated_intelligence = IntegratedIntelligence(
                    file_path=entry.file_path,
                    relative_path=str(Path(entry.file_path).relative_to(self.root_dir)),
                    llm_analysis=entry,
                    static_analysis=static_analysis,
                    integrated_classification=classification_result.primary_classification,
                    integration_confidence=classification_result.confidence_score,
                    confidence_factors=confidence_factors,
                    reorganization_priority=reorganization_priority,
                    final_recommendations=final_recommendations,
                    synthesis_reasoning=classification_result.reasoning
                )

                integrated_results.append(integrated_intelligence)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Integrated {i + 1}/{len(entries)} entries")

            except Exception as e:
                self.logger.error(f"Error integrating entry {i}: {e}")
                continue

        self.logger.info(f"Completed integration of {len(integrated_results)} entries")
        return integrated_results

    def _perform_static_analysis(self, file_path: str) -> StaticAnalysisResult:
        """Perform static analysis on a file"""
        try:
            # Run semantic analysis
            semantic_result = None
            if 'semantic' in self.analyzers:
                semantic_result = self.analyzers['semantic'].analyze_semantics(file_path)

            # Run pattern analysis
            pattern_result = None
            if 'pattern' in self.analyzers:
                pattern_result = self.analyzers['pattern'].detect_patterns(file_path)

            # Run quality analysis
            quality_result = None
            if 'quality' in self.analyzers:
                quality_result = self.analyzers['quality'].analyze_quality(file_path)

            return StaticAnalysisResult(
                semantic=semantic_result,
                pattern=pattern_result,
                quality=quality_result,
                relationship=None  # Could add relationship analysis here
            )

        except Exception as e:
            self.logger.warning(f"Static analysis failed for {file_path}: {e}")
            return StaticAnalysisResult(semantic=None, pattern=None, quality=None, relationship=None)

    def _calculate_confidence_factors(self, llm_entry, static_analysis) -> ConfidenceFactors:
        """Calculate confidence factors from different analysis sources"""
        factors = ConfidenceFactors()

        # LLM confidence
        factors.llm_confidence = llm_entry.confidence_score

        # Semantic confidence
        if static_analysis.semantic:
            factors.semantic_confidence = static_analysis.semantic.get('confidence', 0.5)

        # Pattern confidence
        if static_analysis.pattern:
            factors.pattern_confidence = static_analysis.pattern.get('confidence', 0.5)

        # Quality confidence
        if static_analysis.quality:
            factors.quality_confidence = static_analysis.quality.get('overall_score', 0.5)

        # Agreement confidence (how well analyses agree)
        factors.agreement_confidence = self._calculate_agreement_confidence(llm_entry, static_analysis, factors)

        return factors

    def _calculate_agreement_confidence(self, llm_entry, static_analysis, factors: ConfidenceFactors) -> float:
        """Calculate agreement confidence between different analysis sources"""
        agreements = 0
        total_comparisons = 0

        # Compare LLM classification with static analysis results
        llm_classification = llm_entry.primary_classification.lower()

        if static_analysis.semantic:
            semantic_purpose = static_analysis.semantic.get('primary_purpose', '').lower()
            if semantic_purpose in llm_classification or llm_classification in semantic_purpose:
                agreements += 1
            total_comparisons += 1

        if static_analysis.pattern:
            pattern_categories = [p.get('category', '').lower() for p in static_analysis.pattern.get('patterns', [])]
            if any(cat in llm_classification or llm_classification in cat for cat in pattern_categories):
                agreements += 1
            total_comparisons += 1

        if total_comparisons > 0:
            return agreements / total_comparisons
        return 0.5  # Default if no comparisons possible

    def _dict_to_entry(self, entry_data: Dict) -> Any:
        """Convert dictionary to LLMIntelligenceEntry"""
        # This would need the actual LLMIntelligenceEntry class
        # For now, return a mock object
        class MockEntry:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
                # Set defaults
                self.confidence_score = data.get('confidence_score', 0.5)
                self.primary_classification = data.get('primary_classification', 'unknown')
                self.secondary_classifications = data.get('secondary_classifications', [])
                self.file_path = data.get('file_path', '')

        return MockEntry(entry_data)

    def generate_reorganization_plan(self, integrated_intelligence: List[IntegratedIntelligence],
                                   llm_intelligence_map: Dict[str, Any]) -> Any:
        """Generate a comprehensive reorganization plan"""
        return self.planning_engine.generate_reorganization_plan(
            integrated_intelligence, llm_intelligence_map, ReorganizationPhase
        )

    def save_integration_results(self, integrated_intelligence: List[IntegratedIntelligence],
                               llm_map: Optional[Any], output_file: Path) -> None:
        """Save integration results to file"""
        results = {
            'integration_timestamp': datetime.now().isoformat(),
            'total_entries': len(integrated_intelligence),
            'integrated_entries': [entry.__dict__ if hasattr(entry, '__dict__') else entry for entry in integrated_intelligence],
            'summary': {
                'high_confidence': len([i for i in integrated_intelligence if i.integration_confidence > 0.8]),
                'medium_confidence': len([i for i in integrated_intelligence if 0.6 <= i.integration_confidence <= 0.8]),
                'low_confidence': len([i for i in integrated_intelligence if i.integration_confidence < 0.6])
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Saved integration results to {output_file}")


def main():
    """Main entry point"""
    print("🤖 Intelligence Integration Engine")
    print("=" * 40)
    print("Integrating multiple intelligence sources for comprehensive analysis!")

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Intelligence Integration Engine")
    parser.add_argument("--root", type=str, default=".",
                      help="Root directory to analyze")
    parser.add_argument("--llm-map", type=str, required=True,
                      help="Path to LLM intelligence map")
    parser.add_argument("--output", type=str,
                      help="Output file path")

    args = parser.parse_args()

    # Initialize engine
    root_dir = Path(args.root).resolve()
    engine = IntelligenceIntegrationEngine(root_dir)

    # Load LLM map
    with open(args.llm_map, 'r') as f:
        llm_map = json.load(f)

    # Integrate intelligence
    integrated_intelligence = engine.integrate_intelligence(llm_map)

    # Generate reorganization plan
    plan = engine.generate_reorganization_plan(integrated_intelligence, llm_map)

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = root_dir / f"integrated_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    engine.save_integration_results(integrated_intelligence, None, output_file)

    print(f"✅ Integration completed! Results saved to {output_file}")


if __name__ == "__main__":
    main()
