#!/usr/bin/env python3
"""
Meta-Reorganizer: Intelligence-Driven Codebase Analysis
=======================================================

Leverages your existing intelligence modules for sophisticated code analysis
and reorganization. Uses your semantic analyzers, ML models, and relationship
mappers for granular understanding.

This is a meta-level tool that uses YOUR OWN intelligence infrastructure
to analyze and reorganize your codebase intelligently.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import os
import sys
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
import logging

# Import our modular components
from meta_models import IntelligenceAnalysis, ModuleRelationship
from meta_analysis import IntelligenceAnalysisEngine
from meta_planning import IntelligencePlanningEngine


class IntelligenceDrivenReorganizer:
    """
    Uses your existing intelligence modules to analyze and reorganize code.
    This is the most sophisticated approach possible.
    """

    def __init__(self, root_dir: Path) -> None:
        """Initialize the intelligence-driven reorganizer"""
        self.root_dir = root_dir.resolve()
        self.intelligence_modules = self._discover_intelligence_modules()
        self.exclusions = self._get_exclusions()

        # Setup logging
        self.setup_logging()

        # Initialize our modular components
        self.analysis_engine = IntelligenceAnalysisEngine(self.intelligence_modules, self.logger)
        self.planning_engine = IntelligencePlanningEngine(self.root_dir, self.logger)

        self.logger.info(f"Intelligence-Driven Reorganizer initialized")
        self.logger.info(f"Found {len(self.intelligence_modules)} intelligence modules")

    def setup_logging(self) -> None:
        """Setup logging"""
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
        """Discover available intelligence modules in your codebase"""
        intelligence_modules = {}

        # Look for intelligence modules in known locations
        intelligence_paths = [
            "core/intelligence/analysis",
            "core/intelligence/production/analysis",
            "core/intelligence/monitoring",
            "core/intelligence/predictive",
            "core/intelligence/orchestration",
            "TestMaster/core/intelligence"
        ]

        for path in intelligence_paths:
            full_path = self.root_dir / path
            if full_path.exists():
                for py_file in full_path.glob("*.py"):
                    if py_file.is_file():
                        # Categorize by functionality
                        if "semantic" in py_file.name.lower():
                            intelligence_modules["semantic"] = py_file
                        elif "relationship" in py_file.name.lower():
                            intelligence_modules["relationship"] = py_file
                        elif "pattern" in py_file.name.lower():
                            intelligence_modules["pattern"] = py_file
                        elif "ml" in py_file.name.lower() or "analyzer" in py_file.name.lower():
                            intelligence_modules["ml"] = py_file
                        elif "dependency" in py_file.name.lower():
                            intelligence_modules["dependency"] = py_file

        return intelligence_modules

    def _get_exclusions(self) -> Set[str]:
        """Get the same exclusions as your find_active_python_modules.py"""
        return {
            'archive', 'archives', 'PRODUCTION_PACKAGES',
            'agency-swarm', 'autogen', 'agent-squad', 'agentops',
            'agentscope', 'AgentVerse', 'crewAI', 'CodeGraph',
            'falkordb-py', 'AWorld', 'MetaGPT', 'metagpt',
            'PraisonAI', 'praisonai', 'llama-agents', 'phidata', 'swarms',
            'lagent', 'langgraph-supervisor-py',
            '__pycache__', '.git', 'node_modules', 'htmlcov',
            '.pytest_cache', 'tests', 'test_sessions', 'testmaster_sessions'
        }

    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded"""
        if not path.exists() or not path.is_file() or path.suffix != '.py':
            return True

        try:
            rel_path = path.relative_to(self.root_dir)
            return any(part in self.exclusions for part in rel_path.parts)
        except ValueError:
            return True

    def analyze_with_intelligence_modules(self, file_path: Path) -> IntelligenceAnalysis:
        """
        Use your intelligence modules to analyze a file.
        This is the key innovation - leveraging YOUR OWN sophisticated analysis tools.
        """
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Use our analysis engine to run all intelligence analyses
            semantic_analysis = self.analysis_engine.run_semantic_analysis(file_path, content)
            relationship_analysis = self.analysis_engine.run_relationship_analysis(file_path, content)
            pattern_analysis = self.analysis_engine.run_pattern_analysis(file_path, content)
            ml_analysis = self.analysis_engine.run_ml_analysis(file_path, content)
            dependency_analysis = self.analysis_engine.run_dependency_analysis(file_path, content)

            # Combine analyses to determine category and confidence
            recommended_category, confidence_score, reasoning = self.analysis_engine.combine_analyses(
                semantic_analysis, relationship_analysis, pattern_analysis,
                ml_analysis, dependency_analysis, file_path, content
            )

            return IntelligenceAnalysis(
                file_path=file_path,
                semantic_analysis=semantic_analysis,
                relationship_analysis=relationship_analysis,
                pattern_analysis=pattern_analysis,
                ml_analysis=ml_analysis,
                dependency_analysis=dependency_analysis,
                confidence_score=confidence_score,
                recommended_category=recommended_category,
                reasoning=reasoning
            )

        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path} with intelligence modules: {e}")
            # Fallback to basic analysis
            return self._fallback_analysis(file_path)

    def _fallback_analysis(self, file_path: Path) -> IntelligenceAnalysis:
        """Fallback analysis when intelligence modules fail"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Basic categorization
            category, confidence, reasoning = self.analysis_engine._fallback_categorization(file_path, content, [])

            return IntelligenceAnalysis(
                file_path=file_path,
                semantic_analysis={'category': category, 'confidence': confidence},
                relationship_analysis={'primary_relationship': 'unknown'},
                pattern_analysis={'dominant_pattern': 'generic'},
                ml_analysis={'predicted_category': category},
                dependency_analysis={'primary_domain': 'unknown'},
                confidence_score=confidence,
                recommended_category=category,
                reasoning=reasoning
            )
        except Exception as e:
            self.logger.error(f"Fallback analysis failed for {file_path}: {e}")
            return IntelligenceAnalysis(
                file_path=file_path,
                semantic_analysis={},
                relationship_analysis={},
                pattern_analysis={},
                ml_analysis={},
                dependency_analysis={},
                confidence_score=0.1,
                recommended_category='utilities',
                reasoning=['Fallback analysis due to error']
            )

    def analyze_codebase(self) -> List[IntelligenceAnalysis]:
        """Analyze the entire codebase using intelligence modules"""
        return self.planning_engine.analyze_codebase(self)

    def generate_intelligent_reorganization_plan(self, analyses: List[IntelligenceAnalysis]) -> Dict:
        """Generate reorganization plan based on intelligence analysis"""
        return self.planning_engine.generate_intelligent_reorganization_plan(analyses, self)

    def execute_plan(self, plan: Dict) -> None:
        """Execute the intelligent reorganization plan"""
        self.planning_engine.execute_plan(plan)


def main() -> None:
    """Main function"""
    print("🤖 Intelligent Codebase Reorganizer")
    print("=" * 40)
    print("Using YOUR OWN intelligence modules for sophisticated analysis!")

    # Use current directory as root
    root_dir = Path.cwd()

    reorganizer = IntelligenceDrivenReorganizer(root_dir)

    if not reorganizer.intelligence_modules:
        print("⚠️  No intelligence modules found in your codebase.")
        print("   This tool requires your existing intelligence infrastructure.")
        print("   Expected modules: semantic_analyzer.py, relationship_analyzer.py, etc.")
        sys.exit(1)

    print(f"Found intelligence modules: {list(reorganizer.intelligence_modules.keys())}")

    # Analyze codebase with intelligence
    analyses = reorganizer.analyze_codebase()

    # Generate intelligent reorganization plan
    plan = reorganizer.generate_intelligent_reorganization_plan(analyses)

    # Execute plan (only high-confidence moves)
    reorganizer.execute_plan(plan)

    print("✅ Intelligent reorganization completed!")
    print("Check the logs for detailed analysis results.")


if __name__ == "__main__":
    main()
