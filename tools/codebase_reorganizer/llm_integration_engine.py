#!/usr/bin/env python3
"""
LLM Integration Engine
======================

Integrates multiple intelligence sources (traditional analysis + LLM analysis)
to provide comprehensive reorganization recommendations. Uses LLM to synthesize
insights from various analysis perspectives.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Import our intelligence modules
try:
    from semantic_analyzer import SemanticAnalyzer
    from relationship_analyzer import RelationshipAnalyzer
    from pattern_detector import PatternDetector
    from code_quality_analyzer import CodeQualityAnalyzer
    from llm_intelligence_scanner import LLMIntelligenceScanner, LLMIntelligenceEntry
    HAS_INTELLIGENCE_MODULES = True
except ImportError:
    HAS_INTELLIGENCE_MODULES = False


@dataclass
class IntegratedIntelligence:
    """Integrated intelligence from multiple sources"""
    file_path: str
    traditional_analysis: Dict[str, Any]
    llm_analysis: Dict[str, Any]
    confidence_factors: Dict[str, float]
    integrated_classification: str
    reorganization_priority: int  # 1-10
    integration_confidence: float
    final_recommendations: List[str]
    synthesis_reasoning: str


class LLMIntegrationEngine:
    """
    Integrates LLM analysis with traditional analysis for comprehensive
    reorganization intelligence.
    """

    def __init__(self, root_dir: Path, llm_scanner: Optional[LLMIntelligenceScanner] = None):
        self.root_dir = root_dir.resolve()
        self.llm_scanner = llm_scanner
        self._setup_logging()

        # Initialize traditional analyzers
        self.analyzers = self._initialize_analyzers()

    def _setup_logging(self) -> None:
        """Setup logging"""
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"llm_integration_{timestamp}.log"

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
        """Initialize traditional analysis modules"""
        analyzers = {}

        if HAS_INTELLIGENCE_MODULES:
            try:
                analyzers['semantic'] = SemanticAnalyzer()
                analyzers['relationship'] = RelationshipAnalyzer()
                analyzers['pattern'] = PatternDetector()
                analyzers['quality'] = CodeQualityAnalyzer()
                self.logger.info("Traditional analyzers initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize analyzers: {e}")

        return analyzers

    def integrate_intelligence(self, llm_intelligence_map: Dict[str, Any]) -> List[IntegratedIntelligence]:
        """
        Integrate LLM analysis with traditional analysis.

        Args:
            llm_intelligence_map: The LLM intelligence map from the scanner

        Returns:
            List of integrated intelligence entries
        """
        integrated_results = []

        if 'intelligence_entries' not in llm_intelligence_map:
            self.logger.error("Invalid LLM intelligence map format")
            return integrated_results

        entries = llm_intelligence_map['intelligence_entries']

        for entry_data in entries:
            try:
                # Convert to LLMIntelligenceEntry object
                llm_entry = LLMIntelligenceEntry(**entry_data)

                # Perform traditional analysis
                traditional_analysis = self._perform_traditional_analysis(llm_entry.full_path)

                # Integrate the analyses
                integrated = self._integrate_single_entry(llm_entry, traditional_analysis)

                if integrated:
                    integrated_results.append(integrated)

            except Exception as e:
                self.logger.error(f"Error integrating entry {entry_data.get('full_path', 'unknown')}: {e}")

        self.logger.info(f"Successfully integrated {len(integrated_results)} entries")
        return integrated_results

    def _perform_traditional_analysis(self, file_path: str) -> Dict[str, Any]:
        """Perform traditional analysis using our existing modules"""
        analysis_results = {}

        try:
            path = Path(file_path)
            if not path.exists():
                return analysis_results

            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Run each analyzer
            for name, analyzer in self.analyzers.items():
                try:
                    if name == 'semantic':
                        analysis_results['semantic'] = analyzer.analyze_semantics(content, path)
                    elif name == 'relationship':
                        analysis_results['relationship'] = analyzer.analyze_relationships(content, str(path))
                    elif name == 'pattern':
                        analysis_results['pattern'] = analyzer.detect_patterns(content, path)
                    elif name == 'quality':
                        analysis_results['quality'] = analyzer.analyze_quality(content, path)
                except Exception as e:
                    self.logger.warning(f"{name} analysis failed for {file_path}: {e}")
                    analysis_results[name] = {'error': str(e)}

        except Exception as e:
            self.logger.error(f"Traditional analysis failed for {file_path}: {e}")

        return analysis_results

    def _integrate_single_entry(self, llm_entry: LLMIntelligenceEntry,
                              traditional_analysis: Dict[str, Any]) -> Optional[IntegratedIntelligence]:
        """Integrate a single entry's analyses"""

        # Calculate confidence factors
        confidence_factors = self._calculate_confidence_factors(llm_entry, traditional_analysis)

        # Determine integrated classification
        integrated_classification = self._determine_integrated_classification(
            llm_entry, traditional_analysis, confidence_factors
        )

        # Calculate reorganization priority
        reorganization_priority = self._calculate_reorganization_priority(
            llm_entry, traditional_analysis, confidence_factors
        )

        # Generate integrated recommendations
        final_recommendations = self._generate_integrated_recommendations(
            llm_entry, traditional_analysis, integrated_classification
        )

        # Calculate overall integration confidence
        integration_confidence = sum(confidence_factors.values()) / len(confidence_factors) if confidence_factors else 0.5

        # Generate synthesis reasoning
        synthesis_reasoning = self._generate_synthesis_reasoning(
            llm_entry, traditional_analysis, confidence_factors
        )

        return IntegratedIntelligence(
            file_path=llm_entry.full_path,
            traditional_analysis=traditional_analysis,
            llm_analysis={
                'summary': llm_entry.module_summary,
                'classification': llm_entry.primary_classification,
                'confidence': llm_entry.confidence_score,
                'security': llm_entry.security_implications,
                'complexity': llm_entry.complexity_assessment
            },
            confidence_factors=confidence_factors,
            integrated_classification=integrated_classification,
            reorganization_priority=reorganization_priority,
            integration_confidence=integration_confidence,
            final_recommendations=final_recommendations,
            synthesis_reasoning=synthesis_reasoning
        )

    def _calculate_confidence_factors(self, llm_entry: LLMIntelligenceEntry,
                                    traditional_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence factors for different analysis methods"""

        factors = {}

        # LLM confidence
        factors['llm_confidence'] = llm_entry.confidence_score

        # Traditional analysis confidence
        if 'semantic' in traditional_analysis:
            semantic_conf = traditional_analysis['semantic'].get('semantic_confidence', 0.5)
            factors['semantic_confidence'] = semantic_conf

        if 'pattern' in traditional_analysis:
            pattern_data = traditional_analysis['pattern']
            pattern_conf = pattern_data.get('high_confidence_patterns', 0) / max(pattern_data.get('total_patterns', 1), 1)
            factors['pattern_confidence'] = pattern_conf

        if 'quality' in traditional_analysis:
            quality_data = traditional_analysis['quality']
            quality_conf = quality_data.get('overall_score', 0.5)
            factors['quality_confidence'] = quality_conf

        # Agreement factor (how well analyses agree)
        agreement_score = self._calculate_agreement_score(llm_entry, traditional_analysis)
        factors['agreement_confidence'] = agreement_score

        return factors

    def _calculate_agreement_score(self, llm_entry: LLMIntelligenceEntry,
                                 traditional_analysis: Dict[str, Any]) -> float:
        """Calculate how well different analyses agree"""

        agreement_score = 0.5  # Base score
        factors = 0

        # Compare classifications if available
        if 'semantic' in traditional_analysis:
            semantic_data = traditional_analysis['semantic']
            llm_classification = llm_entry.primary_classification.lower()
            semantic_purpose = semantic_data.get('primary_purpose', '').lower()

            # Simple keyword matching for agreement
            if any(keyword in semantic_purpose for keyword in llm_classification.split('_')):
                agreement_score += 0.2
            factors += 1

        # Compare complexity assessments
        if 'quality' in traditional_analysis:
            quality_data = traditional_analysis['quality']
            llm_complexity = llm_entry.complexity_assessment.lower()

            if 'complex' in llm_complexity and quality_data.get('overall_score', 1.0) < 0.7:
                agreement_score += 0.2
            elif 'simple' in llm_complexity and quality_data.get('overall_score', 0.0) > 0.8:
                agreement_score += 0.2
            factors += 1

        return agreement_score / max(factors, 1) if factors > 0 else 0.5

    def _determine_integrated_classification(self, llm_entry: LLMIntelligenceEntry,
                                           traditional_analysis: Dict[str, Any],
                                           confidence_factors: Dict[str, float]) -> str:
        """Determine the final integrated classification"""

        # Start with LLM classification
        final_classification = llm_entry.primary_classification

        # Check if traditional analysis suggests different classification
        if 'semantic' in traditional_analysis and confidence_factors.get('semantic_confidence', 0) > 0.7:
            semantic_data = traditional_analysis['semantic']
            semantic_purpose = semantic_data.get('primary_purpose', '')

            # Map semantic purposes to classifications
            purpose_to_classification = {
                'data_processing': 'utility',
                'machine_learning': 'intelligence',
                'web_development': 'frontend_dashboard',
                'testing': 'testing',
                'utilities': 'utility',
                'security': 'security',
                'data_analysis': 'intelligence',
                'automation': 'automation'
            }

            semantic_classification = purpose_to_classification.get(semantic_purpose, final_classification)

            # If semantic analysis is more confident, use its classification
            if (confidence_factors.get('semantic_confidence', 0) >
                confidence_factors.get('llm_confidence', 0) * 1.2):
                final_classification = semantic_classification

        return final_classification

    def _calculate_reorganization_priority(self, llm_entry: LLMIntelligenceEntry,
                                         traditional_analysis: Dict[str, Any],
                                         confidence_factors: Dict[str, float]) -> int:
        """Calculate reorganization priority (1-10)"""

        priority = 5  # Base priority

        # LLM factors
        if llm_entry.confidence_score > 0.8:
            priority += 2
        elif llm_entry.confidence_score < 0.5:
            priority -= 1

        # Security concerns increase priority
        if llm_entry.security_implications and 'none' not in llm_entry.security_implications.lower():
            priority += 2

        # High complexity increases priority
        if 'high' in llm_entry.complexity_assessment.lower():
            priority += 1

        # Traditional analysis factors
        if 'quality' in traditional_analysis:
            quality_score = traditional_analysis['quality'].get('overall_score', 0.5)
            if quality_score < 0.6:
                priority += 2  # Poor quality needs attention
            elif quality_score > 0.8:
                priority -= 1  # Good quality can wait

        # Integration confidence
        avg_confidence = sum(confidence_factors.values()) / len(confidence_factors) if confidence_factors else 0.5
        if avg_confidence > 0.8:
            priority += 1
        elif avg_confidence < 0.6:
            priority -= 1

        return max(1, min(10, priority))

    def _generate_integrated_recommendations(self, llm_entry: LLMIntelligenceEntry,
                                           traditional_analysis: Dict[str, Any],
                                           integrated_classification: str) -> List[str]:
        """Generate integrated reorganization recommendations"""

        recommendations = []
        recommendations.extend(llm_entry.reorganization_recommendations)

        # Add traditional analysis insights
        if 'quality' in traditional_analysis:
            quality_data = traditional_analysis['quality']
            if quality_data.get('overall_score', 1.0) < 0.7:
                recommendations.append("Address code quality issues before reorganization")
                quality_issues = quality_data.get('critical_issues', [])
                # Convert quality issues to recommendations (replacing complex comprehension with explicit loop)
                for issue in quality_issues[:2]:
                    recommendations.append(f"Fix: {issue}")

        if 'pattern' in traditional_analysis:
            pattern_data = traditional_analysis['pattern']
            if pattern_data.get('anti-patterns', []):
                recommendations.append("Consider refactoring anti-patterns identified")

        # Add integration-specific recommendations
        if integrated_classification != llm_entry.primary_classification:
            recommendations.append(f"Classification updated from '{llm_entry.primary_classification}' to '{integrated_classification}' based on integrated analysis")

        return recommendations

    def _generate_synthesis_reasoning(self, llm_entry: LLMIntelligenceEntry,
                                    traditional_analysis: Dict[str, Any],
                                    confidence_factors: Dict[str, float]) -> str:
        """Generate reasoning for the synthesis"""

        reasoning_parts = []

        # LLM contribution
        reasoning_parts.append(f"LLM analysis classified as '{llm_entry.primary_classification}' with {llm_entry.confidence_score:.2f} confidence")

        # Traditional analysis contributions
        if 'semantic' in traditional_analysis:
            semantic_purpose = traditional_analysis['semantic'].get('primary_purpose', 'unknown')
            reasoning_parts.append(f"Semantic analysis identified purpose: '{semantic_purpose}'")

        if 'quality' in traditional_analysis:
            quality_score = traditional_analysis['quality'].get('overall_score', 0.5)
            reasoning_parts.append(f"Quality analysis scored {quality_score:.2f} overall")

        # Confidence analysis
        avg_confidence = sum(confidence_factors.values()) / len(confidence_factors) if confidence_factors else 0.5
        reasoning_parts.append(f"Integrated confidence: {avg_confidence:.2f}")

        return " | ".join(reasoning_parts)

    def generate_reorganization_plan(self, integrated_intelligence: List[IntegratedIntelligence]) -> Dict[str, Any]:
        """Generate a comprehensive reorganization plan based on integrated intelligence"""

        # Sort by priority (replacing complex comprehension with explicit loop)
        sorted_intelligence = sorted(integrated_intelligence,
                                   key=lambda x: x.reorganization_priority,
                                   reverse=True)

        # Group by classification
        by_classification = {}
        for intelligence in sorted_intelligence:
            classification = intelligence.integrated_classification
            if classification not in by_classification:
                by_classification[classification] = []
            by_classification[classification].append(intelligence)

        # Generate plan
        plan = {
            'timestamp': datetime.now().isoformat(),
            'total_modules': len(integrated_intelligence),
            # Count high priority modules (replacing complex comprehension with explicit loop)
            high_priority_count = 0
            for i in integrated_intelligence:
                if i.reorganization_priority >= 8:
                    high_priority_count += 1
            'high_priority_modules': high_priority_count,
            'classifications': list(by_classification.keys()),
            'reorganization_phases': self._generate_reorganization_phases(by_classification),
            'estimated_effort': self._estimate_reorganization_effort(sorted_intelligence),
            'risk_assessment': self._assess_reorganization_risks(sorted_intelligence),
            'success_metrics': self._define_success_metrics()
        }

        return plan

    def _create_phase1_high_confidence(self, by_classification: Dict[str, List[IntegratedIntelligence]]) -> List[Dict[str, Any]]:
        """Create Phase 1: High confidence, low risk moves"""
        phase1 = []
        for classification, intelligence_list in by_classification.items():
            # Find high confidence modules with low priority (replacing complex comprehension)
            high_confidence = []
            for i in intelligence_list:
                if i.integration_confidence > 0.8 and i.reorganization_priority < 7:
                    high_confidence.append(i)

            # Add modules to phase (replacing complex comprehension)
            for i in high_confidence:
                phase1.append({
                    'file': i.file_path,
                    'target_classification': classification,
                    'confidence': i.integration_confidence,
                    'reasoning': i.synthesis_reasoning
                })

        if phase1:
            return [{
                'phase': 1,
                'name': 'High Confidence Reorganization',
                'description': 'Move modules with high analysis confidence and low risk',
                'modules': phase1,
                'estimated_time': f"{len(phase1) * 15} minutes",
                'risk_level': 'Low'
            }]
        return []


    def _create_phase2_security_modules(self, by_classification: Dict[str, List[IntegratedIntelligence]]) -> List[Dict[str, Any]]:
        """Create Phase 2: Security and critical modules"""
        phase2 = []
        for classification, intelligence_list in by_classification.items():
            # Find security modules (replacing complex comprehension)
            security_modules = []
            for i in intelligence_list:
                if ('security' in i.integrated_classification or
                    'security' in i.llm_analysis.get('security', '').lower()):
                    security_modules.append(i)

            # Add modules to phase (replacing complex comprehension)
            for i in security_modules:
                phase2.append({
                    'file': i.file_path,
                    'target_classification': classification,
                    'security_notes': i.llm_analysis.get('security', ''),
                    'priority': i.reorganization_priority
                })

        if phase2:
            return [{
                'phase': 2,
                'name': 'Security-Critical Modules',
                'description': 'Handle security-related modules with extra care',
                'modules': phase2,
                'estimated_time': f"{len(phase2) * 30} minutes",
                'risk_level': 'Medium'
            }]
        return []


    def _create_phase3_complex_modules(self, by_classification: Dict[str, List[IntegratedIntelligence]]) -> List[Dict[str, Any]]:
        """Create Phase 3: Complex and high-priority modules"""
        phase3 = []
        for classification, intelligence_list in by_classification.items():
            # Find complex modules (replacing complex comprehension)
            complex_modules = []
            for i in intelligence_list:
                if i.reorganization_priority >= 8:
                    complex_modules.append(i)

            # Add modules to phase (replacing complex comprehension)
            for i in complex_modules:
                phase3.append({
                    'file': i.file_path,
                    'target_classification': classification,
                    'complexity': i.llm_analysis.get('complexity', ''),
                    'priority': i.reorganization_priority
                })

        if phase3:
            return [{
                'phase': 3,
                'name': 'Complex Module Reorganization',
                'description': 'Handle complex, high-priority modules requiring careful planning',
                'modules': phase3,
                'estimated_time': f"{len(phase3) * 45} minutes",
                'risk_level': 'High'
            }]
        return []


    def _generate_reorganization_phases(self, by_classification: Dict[str, List[IntegratedIntelligence]]) -> List[Dict[str, Any]]:
        """Generate phased reorganization approach"""
        phases = []

        # Create each phase using focused functions
        phases.extend(self._create_phase1_high_confidence(by_classification))
        phases.extend(self._create_phase2_security_modules(by_classification))
        phases.extend(self._create_phase3_complex_modules(by_classification))

        return phases

    def _estimate_reorganization_effort(self, sorted_intelligence: List[IntegratedIntelligence]) -> Dict[str, Any]:
        """Estimate the effort required for reorganization"""

        total_modules = len(sorted_intelligence)

        # Count priority levels (replacing complex comprehensions with explicit loops)
        high_priority = 0
        medium_priority = 0
        low_priority = 0

        for i in sorted_intelligence:
            if i.reorganization_priority >= 8:
                high_priority += 1
            elif 6 <= i.reorganization_priority < 8:
                medium_priority += 1
            else:
                low_priority += 1

        # Estimate time in hours
        estimated_hours = (high_priority * 1.5) + (medium_priority * 1.0) + (low_priority * 0.5)

        return {
            'total_modules': total_modules,
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority,
            'estimated_hours': estimated_hours,
            'estimated_days': estimated_hours / 6,  # Assuming 6 hours/day
            'complexity_factor': 'High' if high_priority > total_modules * 0.3 else 'Medium'
        }

    def _assess_reorganization_risks(self, sorted_intelligence: List[IntegratedIntelligence]) -> Dict[str, Any]:
        """Assess risks associated with the reorganization"""

        risks = []

        # Check for security modules
        # Count security modules (replacing complex comprehension with explicit loop)
        security_count = 0
        for i in sorted_intelligence:
            if 'security' in i.integrated_classification:
                security_count += 1

        if security_count > 0:
            risks.append({
                'type': 'Security Risk',
                'severity': 'High',
                'description': f'{security_count} security-related modules require careful handling',
                'mitigation': 'Review security implications before moving modules'
            })

        # Check for low confidence analyses (replacing complex comprehension with explicit loop)
        low_confidence = 0
        for i in sorted_intelligence:
            if i.integration_confidence < 0.6:
                low_confidence += 1

        if low_confidence > 0:
            risks.append({
                'type': 'Analysis Confidence Risk',
                'severity': 'Medium' if low_confidence < 0.3 * len(sorted_intelligence) else 'High',
                'description': f'{low_confidence} modules have low analysis confidence',
                'mitigation': 'Manual review required for low-confidence modules'
            })

        # Check for complex modules (replacing complex comprehension with explicit loop)
        complex_count = 0
        for i in sorted_intelligence:
            if 'high' in i.llm_analysis.get('complexity', '').lower():
                complex_count += 1

        if complex_count > 0:
            risks.append({
                'type': 'Complexity Risk',
                'severity': 'Medium',
                'description': f'{complex_count} complex modules may require additional testing',
                'mitigation': 'Ensure comprehensive testing after reorganization'
            })

        return {
            'risks': risks,
            'overall_risk_level': 'High' if any(r['severity'] == 'High' for r in risks) else 'Medium',
            # Extract recommendations (replacing complex comprehension with explicit loop)
            recommendations = []
            for r in risks:
                recommendations.append(r['mitigation'])
            'recommendations': recommendations
        }

    def _define_success_metrics(self) -> Dict[str, Any]:
        """Define metrics for measuring reorganization success"""

        return {
            'structural_metrics': [
                'Reduction in directory depth',
                'Improved module cohesion',
                'Better separation of concerns'
            ],
            'quality_metrics': [
                'Maintained or improved code quality scores',
                'Reduced complexity in reorganized modules',
                'Better test coverage alignment'
            ],
            'functional_metrics': [
                'All imports still work correctly',
                'No broken functionality',
                'Improved developer experience'
            ],
            'target_goals': [
                'Achieve logical grouping by functionality',
                'Reduce cognitive load for developers',
                'Improve maintainability and extensibility'
            ]
        }


def main():
    """Main function to run the integration engine"""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Integration Engine")
    parser.add_argument("--root", type=str, default=".",
                      help="Root directory")
    parser.add_argument("--llm-map", type=str, required=True,
                      help="Path to LLM intelligence map JSON file")
    parser.add_argument("--output", type=str, default="integrated_intelligence.json",
                      help="Output file for integrated intelligence")

    args = parser.parse_args()

    # Load LLM intelligence map
    try:
        with open(args.llm_map, 'r', encoding='utf-8') as f:
            llm_map = json.load(f)
    except Exception as e:
        print(f"Error loading LLM map: {e}")
        return

    # Initialize integration engine
    root_dir = Path(args.root).resolve()
    engine = LLMIntegrationEngine(root_dir)

    print("🔗 Integrating LLM and Traditional Intelligence...")

    # Perform integration
    integrated_intelligence = engine.integrate_intelligence(llm_map)

    # Generate reorganization plan
    reorganization_plan = engine.generate_reorganization_plan(integrated_intelligence)

    # Save results
    # Convert integrated intelligence to dictionaries (replacing complex comprehension with explicit loop)
    intelligence_list = []
    for intel in integrated_intelligence:
        if hasattr(intel, '__dict__'):
            intelligence_list.append(intel.__dict__)
        else:
            intelligence_list.append(intel)

    output_data = {
        'integrated_intelligence': intelligence_list,
        'reorganization_plan': reorganization_plan,
        'metadata': {
            'integration_timestamp': datetime.now().isoformat(),
            'llm_map_source': args.llm_map,
            'root_directory': str(root_dir)
        }
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Integration completed!")
    print(f"Integrated {len(integrated_intelligence)} modules")
    print(f"Generated reorganization plan with {len(reorganization_plan.get('reorganization_phases', []))} phases")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
