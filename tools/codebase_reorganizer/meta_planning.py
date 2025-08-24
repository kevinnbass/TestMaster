#!/usr/bin/env python3
"""
Meta-Reorganizer Planning Module
=================================

Planning and execution functionality for the intelligence-driven reorganizer system.
Handles reorganization planning, execution, and reporting.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

from meta_models import IntelligenceAnalysis


class IntelligencePlanningEngine:
    """Handles intelligent planning and execution of codebase reorganization"""

    def __init__(self, root_dir: Path, logger):
        """Initialize the planning engine"""
        self.root_dir = root_dir
        self.logger = logger

    def analyze_codebase(self, reorganizer) -> List[IntelligenceAnalysis]:
        """Analyze the entire codebase using intelligence modules"""
        self.logger.info("Starting intelligence-driven codebase analysis...")

        analyses = []

        for root, dirs, files in os.walk(self.root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                (Path(root) / d).match(f"**/{exclusion}/**")
                for exclusion in reorganizer.exclusions
            )]

            for file in files:
                file_path = Path(root) / file
                if not reorganizer.should_exclude(file_path):
                    self.logger.info(f"Analyzing with intelligence modules: {file_path}")
                    analysis = reorganizer.analyze_with_intelligence_modules(file_path)
                    analyses.append(analysis)

                    if len(analyses) % 10 == 0:
                        self.logger.info(f"Analyzed {len(analyses)} files with intelligence modules...")

        self.logger.info(f"Completed analysis of {len(analyses)} files")
        return analyses

    def generate_intelligent_reorganization_plan(self, analyses: List[IntelligenceAnalysis], reorganizer) -> Dict:
        """Generate reorganization plan based on intelligence analysis"""
        plan = {
            'high_confidence_moves': [],
            'medium_confidence_moves': [],
            'low_confidence_moves': [],
            'preserved_directories': [],
            'intelligence_insights': [],
            'summary': {}
        }

        # Group by confidence levels (replacing complex comprehensions with explicit loops)
        high_confidence = []
        medium_confidence = []
        low_confidence = []

        for a in analyses:
            if a.confidence_score >= 0.8:
                high_confidence.append(a)
            elif 0.5 <= a.confidence_score < 0.8:
                medium_confidence.append(a)
            else:
                low_confidence.append(a)

        # Generate moves for high confidence analyses
        for analysis in high_confidence:
            target_path = self._get_intelligent_target_path(analysis)
            if analysis.file_path != target_path:
                plan['high_confidence_moves'].append({
                    'source': str(analysis.file_path),
                    'target': str(target_path),
                    'category': analysis.recommended_category,
                    'confidence': analysis.confidence_score,
                    'intelligence_reasoning': analysis.reasoning,
                    'analysis_data': asdict(analysis)
                })

        # Generate moves for medium confidence analyses
        for analysis in medium_confidence:
            target_path = self._get_intelligent_target_path(analysis)
            if analysis.file_path != target_path:
                plan['medium_confidence_moves'].append({
                    'source': str(analysis.file_path),
                    'target': str(target_path),
                    'category': analysis.recommended_category,
                    'confidence': analysis.confidence_score,
                    'intelligence_reasoning': analysis.reasoning
                })

        # Track low confidence analyses for manual review
        for analysis in low_confidence:
            plan['low_confidence_moves'].append({
                'file': str(analysis.file_path),
                'suggested_category': analysis.recommended_category,
                'confidence': analysis.confidence_score,
                'needs_manual_review': True
            })

        # Generate insights from the intelligence analyses
        plan['intelligence_insights'] = self._extract_insights(analyses)

        # Generate summary
        plan['summary'] = {
            'total_files_analyzed': len(analyses),
            'high_confidence_moves': len(plan['high_confidence_moves']),
            'medium_confidence_moves': len(plan['medium_confidence_moves']),
            'low_confidence_moves': len(plan['low_confidence_moves']),
            'intelligence_modules_used': len(reorganizer.intelligence_modules),
            'average_confidence': sum(a.confidence_score for a in analyses) / len(analyses) if analyses else 0
        }

        return plan

    def _get_intelligent_target_path(self, analysis: IntelligenceAnalysis) -> Path:
        """Determine target path based on intelligence analysis"""
        # Create target directory structure
        category_parts = analysis.recommended_category.split('/')
        target_dir = self.root_dir / 'intelligently_organized_codebase'
        for part in category_parts:
            target_dir = target_dir / part

        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create target file path
        return target_dir / analysis.file_path.name

    def _extract_insights(self, analyses: List[IntelligenceAnalysis]) -> List[Dict]:
        """Extract insights from the intelligence analyses"""
        insights = []

        # Find patterns in categorization decisions
        category_counts = {}
        for analysis in analyses:
            category = analysis.recommended_category
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1

        # Add insights about category distribution
        if category_counts:
            most_common = max(category_counts.items(), key=lambda x: x[1])
            insights.append({
                'type': 'category_distribution',
                'insight': f"Most common category: {most_common[0]} ({most_common[1]} files)",
                'data': category_counts
            })

        # Find files with conflicting analyses
        for analysis in analyses:
            if analysis.confidence_score < 0.3:
                insights.append({
                    'type': 'low_confidence',
                    'insight': f"Low confidence analysis for {analysis.file_path.name}",
                    'file': str(analysis.file_path),
                    'confidence': analysis.confidence_score
                })

        # Look for semantic clusters
        semantic_clusters = self._find_semantic_clusters(analyses)
        if semantic_clusters:
            insights.append({
                'type': 'semantic_clusters',
                'insight': f"Found {len(semantic_clusters)} semantic clusters",
                'data': semantic_clusters
            })

        return insights

    def _find_semantic_clusters(self, analyses: List[IntelligenceAnalysis]) -> List[Dict]:
        """Find clusters of semantically related files"""
        # This would use your semantic analysis results to find clusters
        # For now, return empty list as placeholder
        return []

    def execute_plan(self, plan: Dict) -> None:
        """Execute the intelligent reorganization plan"""
        self.logger.info("Executing intelligent reorganization plan...")

        # Only execute high-confidence moves automatically
        if plan['high_confidence_moves']:
            self.logger.info(f"Executing {len(plan['high_confidence_moves'])} high-confidence moves")

            for move in plan['high_confidence_moves']:
                try:
                    source = Path(move['source'])
                    target = Path(move['target'])

                    target.parent.mkdir(parents=True, exist_ok=True)

                    if not source.exists():
                        self.logger.warning(f"Source file does not exist: {source}")
                        continue

                    # Move the file
                    shutil.move(source, target)
                    self.logger.info(f"✅ Moved: {source} -> {target}")

                except Exception as e:
                    self.logger.error(f"❌ Failed to move {move['source']}: {e}")

        # Report on what was done
        self._print_intelligent_summary(plan)

    def _print_intelligent_summary(self, plan: Dict) -> None:
        """Print intelligent reorganization summary"""
        print("\n" + "="*80)
        print("🤖 INTELLIGENT REORGANIZATION RESULTS")
        print("="*80)

        summary = plan['summary']
        print("📊 SUMMARY:")
        print(f"   Files analyzed with intelligence: {summary['total_files_analyzed']}")
        print(f"   High-confidence moves: {summary['high_confidence_moves']}")
        print(f"   Medium-confidence moves: {summary['medium_confidence_moves']}")
        print(f"   Low-confidence items: {summary['low_confidence_moves']}")
        print(f"   Intelligence modules used: {summary['intelligence_modules_used']}")
        print(".3f")

        print("🎯 INSIGHTS FROM YOUR INTELLIGENCE MODULES:")
        for insight in plan['intelligence_insights'][:5]:
            print(f"   💡 {insight['insight']}")

        print("✅ HIGH-CONFIDENCE MOVES EXECUTED:")
        for move in plan['high_confidence_moves'][:10]:
            print(".2f")
            print(f"      Based on: {', '.join(move['intelligence_reasoning'][:2])}")

        if len(plan['high_confidence_moves']) > 10:
            print(f"   ... and {len(plan['high_confidence_moves']) - 10} more high-confidence moves")

        print("🔄 MEDIUM-CONFIDENCE ITEMS (REVIEW MANUALLY):")
        for move in plan['medium_confidence_moves'][:5]:
            print(".2f")
        if len(plan['medium_confidence_moves']) > 5:
            print(f"   ... and {len(plan['medium_confidence_moves']) - 5} more for review")

        print("📈 INTELLIGENCE MODULES LEVERAGED:")
        for module_type, module_path in getattr(self, 'intelligence_modules', {}).items():
            print(f"   🧠 {module_type}: {module_path.name}")
        print("🎉 This reorganization was driven by YOUR OWN intelligence infrastructure!")
        print("   Your semantic analyzers, ML models, and relationship mappers provided")
        print("   the granular understanding that made this possible.")
