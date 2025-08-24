#!/usr/bin/env python3
"""
Intelligence-Enhanced Codebase Reorganizer
============================================

Leverages existing intelligence modules from the codebase for sophisticated
code analysis and reorganization. Integrates:

- AI Detection Engine (pattern matching & ML analysis)
- Cohesion Analysis (module quality assessment)
- Coupling Analysis (dependency relationships)
- Complexity Analysis (code quality metrics)
- Semantic Analysis (code understanding)

This creates a highly intelligent reorganization system that makes
data-driven decisions based on multiple analysis perspectives.
"""

import os
import sys
import ast
import json
import shutil
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import re
from collections import defaultdict
import hashlib


@dataclass
class IntelligenceAnalysis:
    """Comprehensive analysis from multiple intelligence sources"""
    file_path: Path
    ai_detection_score: float  # From AI detection engine
    cohesion_metrics: Dict[str, Any]  # From cohesion analyzer
    coupling_metrics: Dict[str, Any]  # From coupling analyzer
    complexity_metrics: Dict[str, Any]  # From complexity analyzer
    semantic_analysis: Dict[str, Any]  # From semantic analyzer
    pattern_matches: List[Dict[str, Any]]  # From pattern detector
    overall_confidence: float
    recommended_actions: List[str]
    reasoning: List[str]


@dataclass
class IntelligenceReorganizationPlan:
    """Intelligent reorganization plan based on multiple analyses"""
    analyses: List[IntelligenceAnalysis]
    high_confidence_moves: List[Dict[str, Any]]
    medium_confidence_moves: List[Dict[str, Any]]
    low_confidence_moves: List[Dict[str, Any]]
    summary: Dict[str, Any]
    intelligence_insights: List[Dict[str, Any]]
    recommendations: List[str]


class IntelligenceReorganizer:
    """
    Advanced reorganizer that leverages multiple intelligence modules
    for sophisticated codebase analysis and reorganization.
    """

    def __init__(self, root_dir: Path, mode: str = "preview") -> None:
        """Initialize the intelligence-enhanced reorganizer"""
        self.root_dir = root_dir.resolve()
        self.mode = mode
        self.exclusions = self._get_exclusions()

        # Intelligence module paths
        self.intelligence_modules = self._discover_intelligence_modules()

        # Setup logging
        self._setup_logging()

        self.logger.info("Intelligence-Enhanced Reorganizer initialized")
        self.logger.info(f"Found {len(self.intelligence_modules)} intelligence modules")

    def _setup_logging(self) -> None:
        """Setup comprehensive logging"""
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"intelligence_reorganization_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _discover_intelligence_modules(self) -> Dict[str, Path]:
        """Discover available intelligence modules in the broader codebase"""
        intelligence_modules = {}

        # Look for intelligence modules in various locations
        search_paths = [
            "ai_detector/src/core/detection",
            "ai_detector/src/core/quality",
            "ai_detector/src/core/patterns",
            "ai_detector/src/core/monitoring",
            "src/core/intelligence",
            "core/intelligence/analysis",
            "core/intelligence/production/analysis",
            "core/intelligence/monitoring",
            "core/intelligence/predictive",
            "core/intelligence/orchestration"
        ]

        for path_str in search_paths:
            full_path = self.root_dir / path_str
            if full_path.exists():
                self._scan_directory_for_modules(full_path, intelligence_modules)

        return intelligence_modules

    def _scan_directory_for_modules(self, directory: Path, modules: Dict[str, Path]) -> None:
        """Scan a directory for intelligence modules"""
        for py_file in directory.glob("*.py"):
            if py_file.is_file() and not py_file.name.startswith('test_'):
                # Categorize by functionality
                file_name = py_file.name.lower()
                if "cohesion" in file_name or "cohesion" in py_file.read_text().lower():
                    modules["cohesion"] = py_file
                elif "coupling" in file_name or "coupling" in py_file.read_text().lower():
                    modules["coupling"] = py_file
                elif "complexity" in file_name or "complexity" in py_file.read_text().lower():
                    modules["complexity"] = py_file
                elif "pattern" in file_name or "pattern" in py_file.read_text().lower():
                    modules["pattern"] = py_file
                elif "semantic" in file_name or "semantic" in py_file.read_text().lower():
                    modules["semantic"] = py_file
                elif "detection" in file_name or "detector" in file_name:
                    modules["detection"] = py_file
                elif "analysis" in file_name or "analyzer" in file_name:
                    modules["analysis"] = py_file

    def _get_exclusions(self) -> Set[str]:
        """Get exclusion patterns for files we shouldn't reorganize"""
        return {
            'archive', 'archives', 'PRODUCTION_PACKAGES',
            'agency-swarm', 'autogen', 'agent-squad', 'agentscope',
            'agentscope', 'AgentVerse', 'crewAI', 'CodeGraph',
            'falkordb-py', 'AWorld', 'MetaGPT', 'metagpt',
            'PraisonAI', 'praisonai', 'llama-agents', 'phidata', 'swarms',
            'lagent', 'langgraph-supervisor-py',
            '__pycache__', '.git', 'node_modules', 'htmlcov',
            '.pytest_cache', 'tests', 'test_sessions', 'testmaster_sessions',
            'tools', 'codebase_reorganizer'  # Don't reorganize our own tools
        }

    def should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded from reorganization"""
        if not path.exists() or not path.is_file() or path.suffix != '.py':
            return True

        try:
            rel_path = path.relative_to(self.root_dir)
            return any(part in self.exclusions for part in rel_path.parts)
        except ValueError:
            return True

    def analyze_codebase_with_intelligence(self) -> List[IntelligenceAnalysis]:
        """
        Use multiple intelligence modules to analyze the codebase
        This is the core innovation - leveraging your existing sophisticated analysis tools
        """
        analyses = []

        # Find all Python files to analyze
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not any(excl in d for excl in self.exclusions)]
            for file in files:
                if file.endswith('.py') and not any(excl in file for excl in ['test_', 'setup.py']):
                    file_path = Path(root) / file
                    if not self.should_exclude(file_path):
                        python_files.append(file_path)

        self.logger.info(f"Found {len(python_files)} Python files to analyze with intelligence")

        # Analyze each file with multiple intelligence modules
        for file_path in python_files[:50]:  # Limit for performance
            try:
                analysis = self._analyze_file_with_intelligence(file_path)
                if analysis:
                    analyses.append(analysis)
            except Exception as e:
                self.logger.warning(f"Error analyzing {file_path}: {e}")

        return analyses

    def _analyze_file_with_intelligence(self, file_path: Path) -> Optional[IntelligenceAnalysis]:
        """Analyze a single file using multiple intelligence modules"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Run multiple intelligence analyses
            ai_detection_score = self._run_ai_detection_analysis(file_path, content)
            cohesion_metrics = self._run_cohesion_analysis(file_path, content)
            coupling_metrics = self._run_coupling_analysis(file_path, content)
            complexity_metrics = self._run_complexity_analysis(file_path, content)
            semantic_analysis = self._run_semantic_analysis(file_path, content)
            pattern_matches = self._run_pattern_analysis(file_path, content)

            # Combine analyses for overall assessment
            overall_confidence, recommended_actions, reasoning = self._combine_intelligence_analyses(
                ai_detection_score, cohesion_metrics, coupling_metrics,
                complexity_metrics, semantic_analysis, pattern_matches, file_path, content
            )

            return IntelligenceAnalysis(
                file_path=file_path,
                ai_detection_score=ai_detection_score,
                cohesion_metrics=cohesion_metrics,
                coupling_metrics=coupling_metrics,
                complexity_metrics=complexity_metrics,
                semantic_analysis=semantic_analysis,
                pattern_matches=pattern_matches,
                overall_confidence=overall_confidence,
                recommended_actions=recommended_actions,
                reasoning=reasoning
            )

        except Exception as e:
            self.logger.debug(f"Could not analyze {file_path} with intelligence: {e}")
            return None

    def _run_ai_detection_analysis(self, file_path: Path, content: str) -> float:
        """Use AI detection engine to analyze code patterns"""
        try:
            if "detection" in self.intelligence_modules:
                detector_module = self.intelligence_modules["detection"]

                spec = importlib.util.spec_from_file_location("detector", detector_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Try to use the detection engine
                    if hasattr(module, 'UnifiedDetector'):
                        detector = module.UnifiedDetector()
                        result = detector.detect(content)
                        return result.confidence if hasattr(result, 'confidence') else 0.5
                    elif hasattr(module, 'fast_ml_detector'):
                        return module.fast_ml_detector(content)
        except Exception as e:
            self.logger.debug(f"AI detection failed: {e}")

        return 0.5  # Neutral score

    def _run_cohesion_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Use cohesion analyzer to assess module quality"""
        try:
            if "cohesion" in self.intelligence_modules:
                cohesion_module = self.intelligence_modules["cohesion"]

                spec = importlib.util.spec_from_file_location("cohesion", cohesion_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, 'CohesionAnalyzer'):
                        analyzer = module.CohesionAnalyzer()
                        metrics = analyzer._analyze_file_cohesion(file_path, file_path.name)
                        if metrics:
                            return {
                                'cohesion_score': metrics.cohesion_score,
                                'method_interactions': metrics.method_interactions,
                                'shared_data': metrics.shared_data,
                                'suggestions': metrics.suggestions
                            }
        except Exception as e:
            self.logger.debug(f"Cohesion analysis failed: {e}")

        return {'cohesion_score': 0.5, 'suggestions': []}

    def _run_coupling_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Use coupling analyzer to assess dependencies"""
        try:
            if "coupling" in self.intelligence_modules:
                coupling_module = self.intelligence_modules["coupling"]

                spec = importlib.util.spec_from_file_location("coupling", coupling_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Try to use coupling analysis
                    if hasattr(module, 'CouplingAnalyzer'):
                        analyzer = module.CouplingAnalyzer()
                        result = analyzer.analyze_file_coupling(file_path)
                        return result if result else {}
        except Exception as e:
            self.logger.debug(f"Coupling analysis failed: {e}")

        return {}

    def _run_complexity_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Use complexity analyzer to assess code quality"""
        try:
            if "complexity" in self.intelligence_modules:
                complexity_module = self.intelligence_modules["complexity"]

                spec = importlib.util.spec_from_file_location("complexity", complexity_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Try to use complexity analysis
                    if hasattr(module, 'ComplexityReducer'):
                        analyzer = module.ComplexityReducer()
                        result = analyzer.analyze_complexity(content)
                        return result if result else {}
        except Exception as e:
            self.logger.debug(f"Complexity analysis failed: {e}")

        return {}

    def _run_semantic_analysis(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Use semantic analyzer to understand code meaning"""
        try:
            if "semantic" in self.intelligence_modules:
                semantic_module = self.intelligence_modules["semantic"]

                spec = importlib.util.spec_from_file_location("semantic", semantic_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Try to use semantic analysis
                    if hasattr(module, 'SemanticAnalyzer'):
                        analyzer = module.SemanticAnalyzer()
                        result = analyzer.analyze_semantics(content)
                        return result if result else {}
        except Exception as e:
            self.logger.debug(f"Semantic analysis failed: {e}")

        return {}

    def _run_pattern_analysis(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Use pattern detector to find code patterns"""
        try:
            if "pattern" in self.intelligence_modules:
                pattern_module = self.intelligence_modules["pattern"]

                spec = importlib.util.spec_from_file_location("pattern", pattern_module)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Try to use pattern detection
                    if hasattr(module, 'PatternEngine'):
                        engine = module.PatternEngine()
                        patterns = engine.detect_patterns(content)
                        return patterns if patterns else []
        except Exception as e:
            self.logger.debug(f"Pattern analysis failed: {e}")

        return []

    def _combine_intelligence_analyses(self, ai_score: float, cohesion: Dict[str, Any],
                                     coupling: Dict[str, Any], complexity: Dict[str, Any],
                                     semantic: Dict[str, Any], patterns: List[Dict[str, Any]],
                                     file_path: Path, content: str) -> Tuple[float, List[str], List[str]]:
        """Combine multiple intelligence analyses into actionable recommendations"""

        confidence_factors = []
        reasoning = []
        actions = []

        # AI detection confidence
        if ai_score > 0.7:
            confidence_factors.append(0.9)
            reasoning.append(f"High AI pattern confidence ({ai_score:.2f})")
            actions.append("High-confidence reorganization candidate")
        elif ai_score > 0.5:
            confidence_factors.append(0.7)
            reasoning.append(f"Medium AI pattern confidence ({ai_score:.2f})")
            actions.append("Medium-confidence reorganization")
        else:
            confidence_factors.append(0.5)
            reasoning.append(f"Low AI pattern confidence ({ai_score:.2f})")

        # Cohesion analysis
        cohesion_score = cohesion.get('cohesion_score', 0.5)
        if cohesion_score < 0.4:
            confidence_factors.append(0.8)
            reasoning.append(f"Low cohesion detected ({cohesion_score:.2f})")
            actions.append("High-priority refactoring needed")
            actions.extend(cohesion.get('suggestions', []))
        elif cohesion_score > 0.7:
            confidence_factors.append(0.9)
            reasoning.append(f"High cohesion maintained ({cohesion_score:.2f})")
            actions.append("Preserve current organization")

        # Complexity analysis
        if complexity and complexity.get('complexity_score', 0) > 10:
            confidence_factors.append(0.7)
            reasoning.append("High complexity detected")
            actions.append("Consider complexity reduction")

        # Pattern analysis
        if patterns:
            confidence_factors.append(0.6)
            reasoning.append(f"Found {len(patterns)} code patterns")
            actions.append("Patterns suggest specific categorization")

        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

        return overall_confidence, actions, reasoning

    def generate_intelligent_reorganization_plan(self, analyses: List[IntelligenceAnalysis]) -> IntelligenceReorganizationPlan:
        """Generate a reorganization plan based on intelligence analysis"""

        high_confidence_moves = []
        medium_confidence_moves = []
        low_confidence_moves = []
        intelligence_insights = []

        # Process each analysis
        for analysis in analyses:
            if analysis.overall_confidence >= 0.8:
                high_confidence_moves.extend(self._create_moves_from_analysis(analysis, "high"))
            elif analysis.overall_confidence >= 0.6:
                medium_confidence_moves.extend(self._create_moves_from_analysis(analysis, "medium"))
            else:
                low_confidence_moves.extend(self._create_moves_from_analysis(analysis, "low"))

            # Extract intelligence insights
            intelligence_insights.append({
                'file': str(analysis.file_path),
                'confidence': analysis.overall_confidence,
                'insights': analysis.reasoning,
                'ai_detection': analysis.ai_detection_score,
                'cohesion': analysis.cohesion_metrics.get('cohesion_score', 0),
                'patterns_found': len(analysis.pattern_matches)
            })

        # Generate summary statistics
        total_analyses = len(analyses)
        avg_confidence = sum(a.overall_confidence for a in analyses) / total_analyses if analyses else 0
        high_conf_count = len(high_confidence_moves)
        medium_conf_count = len(medium_confidence_moves)
        low_conf_count = len(low_confidence_moves)

        summary = {
            'total_files_analyzed': total_analyses,
            'average_confidence': avg_confidence,
            'high_confidence_moves': high_conf_count,
            'medium_confidence_moves': medium_conf_count,
            'low_confidence_moves': low_conf_count,
            'intelligence_modules_used': len(self.intelligence_modules),
            'intelligence_confidence': avg_confidence
        }

        # Generate recommendations
        recommendations = self._generate_intelligence_recommendations(summary, intelligence_insights)

        return IntelligenceReorganizationPlan(
            analyses=analyses,
            high_confidence_moves=high_confidence_moves,
            medium_confidence_moves=medium_confidence_moves,
            low_confidence_moves=low_confidence_moves,
            summary=summary,
            intelligence_insights=intelligence_insights,
            recommendations=recommendations
        )

    def _create_moves_from_analysis(self, analysis: IntelligenceAnalysis, confidence_level: str) -> List[Dict[str, Any]]:
        """Create reorganization moves from intelligence analysis"""
        moves = []

        # Determine target directory based on analysis
        target_dir = self._determine_target_directory(analysis)

        if target_dir:
            moves.append({
                'source': str(analysis.file_path),
                'target': target_dir,
                'confidence': analysis.overall_confidence,
                'confidence_level': confidence_level,
                'intelligence_reasoning': analysis.reasoning,
                'ai_detection_score': analysis.ai_detection_score,
                'cohesion_score': analysis.cohesion_metrics.get('cohesion_score', 0),
                'pattern_count': len(analysis.pattern_matches)
            })

        return moves

    def _determine_target_directory(self, analysis: IntelligenceAnalysis) -> Optional[str]:
        """Determine the best target directory for a file based on intelligence analysis"""

        # Use AI detection to categorize
        if analysis.ai_detection_score > 0.7:
            return "src/core/ai_generated"
        elif analysis.ai_detection_score < 0.3:
            return "src/core/human_generated"

        # Use cohesion for organization
        cohesion_score = analysis.cohesion_metrics.get('cohesion_score', 0.5)
        if cohesion_score < 0.4:
            return "src/core/needs_refactoring"
        elif cohesion_score > 0.8:
            return "src/core/well_organized"

        # Use pattern analysis for categorization
        if analysis.pattern_matches:
            pattern_types = set(p.get('type', '') for p in analysis.pattern_matches)
            if 'utility' in pattern_types:
                return "src/core/utilities"
            elif 'data_processing' in pattern_types:
                return "src/core/data_processing"
            elif 'api' in pattern_types:
                return "src/core/api"
            elif 'testing' in pattern_types:
                return "src/core/testing"

        return "src/core/uncategorized"

    def _generate_intelligence_recommendations(self, summary: Dict[str, Any],
                                             insights: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on intelligence analysis"""
        recommendations = []

        # Overall confidence assessment
        avg_confidence = summary.get('average_confidence', 0)
        if avg_confidence > 0.8:
            recommendations.append("HIGH CONFIDENCE: Intelligence analysis shows clear reorganization patterns")
        elif avg_confidence > 0.6:
            recommendations.append("MEDIUM CONFIDENCE: Proceed with caution, review recommendations manually")
        else:
            recommendations.append("LOW CONFIDENCE: Consider manual review before applying changes")

        # Module utilization
        modules_used = summary.get('intelligence_modules_used', 0)
        if modules_used >= 3:
            recommendations.append(f"Using {modules_used} intelligence modules for comprehensive analysis")
        elif modules_used == 0:
            recommendations.append("No intelligence modules found - falling back to basic analysis")

        # High confidence moves
        high_conf_moves = summary.get('high_confidence_moves', 0)
        if high_conf_moves > 10:
            recommendations.append(f"Found {high_conf_moves} high-confidence moves - safe to automate")
        elif high_conf_moves > 0:
            recommendations.append(f"Found {high_conf_moves} high-confidence moves - review individually")

        # Intelligence insights
        if insights:
            top_insights = sorted(insights, key=lambda x: x['confidence'], reverse=True)[:3]
            recommendations.append("Key insights from analysis:")
            for insight in top_insights:
                recommendations.append(f"  • {insight['file']}: {insight['insights'][0] if insight['insights'] else 'Analysis complete'}")

        return recommendations

    def execute_plan(self, plan: IntelligenceReorganizationPlan) -> None:
        """Execute the intelligent reorganization plan"""
        if self.mode == 'preview':
            self._print_intelligent_plan(plan)
            return

        # Execute high-confidence moves automatically
        if plan.high_confidence_moves:
            self._execute_moves(plan.high_confidence_moves, "high-confidence")

        # For medium confidence, ask user or provide warnings
        if plan.medium_confidence_moves:
            self._handle_medium_confidence_moves(plan.medium_confidence_moves)

        # Low confidence moves require manual review
        if plan.low_confidence_moves:
            self._handle_low_confidence_moves(plan.low_confidence_moves)

    def _print_intelligent_plan(self, plan: IntelligenceReorganizationPlan) -> None:
        """Print the intelligent reorganization plan"""
        print("\n🤖 INTELLIGENT REORGANIZATION PLAN")
        print("=" * 60)

        summary = plan.summary
        print("\n📊 SUMMARY:")
        print(f"   Files analyzed with intelligence: {summary['total_files_analyzed']}")
        print(f"   High-confidence moves: {summary['high_confidence_moves']}")
        print(f"   Medium-confidence moves: {summary['medium_confidence_moves']}")
        print(f"   Low-confidence items: {summary['low_confidence_moves']}")
        print(f"   Intelligence modules used: {summary['intelligence_modules_used']}")
        print(".3f")
        print("🎯 INSIGHTS FROM YOUR INTELLIGENCE MODULES:")
        for insight in plan.intelligence_insights[:5]:
            print(f"   💡 {insight['file']}: Confidence {insight['confidence']:.2f}")
            if insight['insights']:
                print(f"      {insight['insights'][0]}")

        print("\n✅ HIGH-CONFIDENCE MOVES EXECUTED:")
        for move in plan.high_confidence_moves[:10]:
            print(".2f")
            print(f"      Based on: {', '.join(move['intelligence_reasoning'][:2])}")

        if len(plan.high_confidence_moves) > 10:
            print(f"   ... and {len(plan.high_confidence_moves) - 10} more high-confidence moves")

        print("\n🔄 MEDIUM-CONFIDENCE ITEMS (REVIEW MANUALLY):")
        for move in plan.medium_confidence_moves[:5]:
            print(".2f")
        if len(plan.medium_confidence_moves) > 5:
            print(f"   ... and {len(plan.medium_confidence_moves) - 5} more for review")

        print("\n📈 INTELLIGENCE MODULES LEVERAGED:")
        for module_type, module_path in self.intelligence_modules.items():
            print(f"   🧠 {module_type}: {module_path.name}")

        print("\n🎉 This reorganization was driven by YOUR OWN intelligence infrastructure!")
        print("   Your AI detectors, cohesion analyzers, and pattern matchers provided")
        print("   the sophisticated understanding that made this possible.")

    def _execute_moves(self, moves: List[Dict[str, Any]], confidence_level: str) -> None:
        """Execute reorganization moves"""
        for move in moves:
            try:
                source_path = Path(move['source'])
                target_path = Path(move['target']) / source_path.name

                if self.mode != 'preview':
                    # Create target directory
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Move file
                    shutil.move(str(source_path), str(target_path))

                self.logger.info(f"{confidence_level.upper()} MOVE: {source_path} -> {target_path}")

            except Exception as e:
                self.logger.error(f"Error moving {move['source']}: {e}")

    def _handle_medium_confidence_moves(self, moves: List[Dict[str, Any]]) -> None:
        """Handle medium confidence moves (may require user approval)"""
        if self.mode == 'interactive':
            for move in moves:
                print(f"Medium confidence move: {move['source']} -> {move['target']}")
                print(f"Intelligence reasoning: {', '.join(move['intelligence_reasoning'])}")
                response = input("Execute this move? (y/n): ")
                if response.lower() == 'y':
                    self._execute_moves([move], "medium-confidence")
        else:
            self.logger.info(f"Found {len(moves)} medium-confidence moves - manual review recommended")

    def _handle_low_confidence_moves(self, moves: List[Dict[str, Any]]) -> None:
        """Handle low confidence moves (require manual review)"""
        self.logger.info(f"Found {len(moves)} low-confidence moves - manual review required")
        for move in moves[:5]:  # Show first 5
            self.logger.info(f"Low confidence: {move['source']} -> {move['target']}")
            self.logger.info(f"Reasoning: {', '.join(move['intelligence_reasoning'])}")

    def fallback_basic_reorganization(self) -> None:
        """Fallback to basic reorganization if intelligence analysis fails"""
        self.logger.info("Falling back to basic reorganization")

        # Use the existing basic reorganizer
        try:
            from reorganizer import CodebaseReorganizer
            basic_reorganizer = CodebaseReorganizer(self.root_dir, self.mode)
            basic_reorganizer.run_reorganization()
        except Exception as e:
            self.logger.error(f"Basic reorganization also failed: {e}")


def main() -> None:
    """Main entry point for the intelligence-enhanced reorganizer"""
    print("🤖 Intelligence-Enhanced Codebase Reorganizer")
    print("=" * 50)
    print("Leveraging your existing AI detection, cohesion analysis,")
    print("pattern matching, and semantic analysis capabilities!")

    # Use current directory as root
    root_dir = Path.cwd()

    reorganizer = IntelligenceReorganizer(root_dir, "preview")

    # Run intelligent analysis
    try:
        analyses = reorganizer.analyze_codebase_with_intelligence()
        if analyses:
            plan = reorganizer.generate_intelligent_reorganization_plan(analyses)
            reorganizer.execute_plan(plan)
        else:
            print("No analyses generated - falling back to basic reorganization")
            reorganizer.fallback_basic_reorganization()

    except Exception as e:
        print(f"Error in intelligent reorganization: {e}")
        print("Falling back to basic reorganization...")
        reorganizer.fallback_basic_reorganization()


if __name__ == "__main__":
    main()
