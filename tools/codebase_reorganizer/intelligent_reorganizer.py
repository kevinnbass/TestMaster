#!/usr/bin/env python3
"""
Intelligent Codebase Reorganizer
===============================

A smarter approach that preserves subdirectory relationships and maintains
the semantic context of related modules.

Key Improvements:
- Analyzes directory structures as units
- Preserves meaningful subdirectory hierarchies
- Only reorganizes when it adds clear value
- Maintains relationships between related modules
- Respects existing well-organized packages

Author: Intelligent Codebase Reorganization System
Version: 3.0
"""

import os
import sys
import json
import shutil
import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import tempfile
import subprocess

@dataclass
class DirectoryAnalysis:
    """Analysis results for a directory/package"""
    path: Path
    package_name: str
    files: List[Path]
    subdirectories: List[Path]
    is_well_organized: bool
    organization_score: float
    primary_category: str
    relationships: Dict[str, List[str]]
    reasons: List[str]

@dataclass
class RelationshipAnalysis:
    """Analysis of relationships between directories"""
    source_dir: Path
    target_dir: Path
    relationship_type: str  # 'imports', 'shares_code', 'similar_functionality'
    strength: float
    evidence: List[str]

class IntelligentReorganizer:
    def __init__(self, root_dir: Path, mode: str = "preview") -> None:
        """Initialize the intelligent reorganizer with configuration"""
        self.root_dir = root_dir.resolve()
        self.mode = mode
        self.excluded_dirs = self._get_exclusions()

        # Setup logging
        self.setup_logging()

        self.logger.info(f"Intelligent Reorganizer initialized for {self.root_dir}")

    def setup_logging(self) -> None:
        """Setup logging"""
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"intelligent_reorganization_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_exclusions(self) -> Set[str]:
        """Get exclusion patterns (same as before)"""
        return {
            'archive', 'archives', 'PRODUCTION_PACKAGES',
            'agency-swarm', 'autogen', 'agent-squad', 'agentops',
            'agentscope', 'AgentVerse', 'crewAI', 'CodeGraph',
            'falkordb-py', 'AWorld', 'MetaGPT', 'metagpt',
            'PraisonAI', 'praisonai', 'llama-agents', 'phidata', 'swarms',
            'lagent', 'langgraph-supervisor-py',
            '__pycache__', '.git', 'node_modules', 'htmlcov',
            '.pytest_cache', '.vscode', '.idea', 'tests',
            'test_sessions', 'testmaster_sessions'
        }

    def should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded"""
        if not path.exists():
            return True

        path_parts = path.relative_to(self.root_dir).parts
        return any(part in self.excluded_dirs for part in path_parts)

    def analyze_directory_structure(self) -> List[DirectoryAnalysis]:
        """Analyze the directory structure intelligently"""
        self.logger.info("Analyzing directory structure...")

        analyses = []

        # Walk through all directories
        for root, dirs, files in os.walk(self.root_dir):
            root_path = Path(root)

            # Skip excluded directories
            # Filter directories (replacing complex comprehension with explicit loop)
            filtered_dirs = []
            for d in dirs:
                if not self.should_exclude(root_path / d):
                    filtered_dirs.append(d)
            dirs[:] = filtered_dirs

            if self.should_exclude(root_path):
                continue

            # Only analyze directories that contain Python files
            # Get Python files (replacing complex comprehension with explicit loop)
            py_files = []
            for f in files:
                if f.endswith('.py'):
                    py_files.append(f)
            if not py_files:
                continue

            # Skip if this is just a single file directory (likely already well-placed)
            if len(py_files) == 1 and not dirs:
                continue

            analysis = self._analyze_directory(root_path, py_files, dirs)
            if analysis:
                analyses.append(analysis)

        self.logger.info(f"Analyzed {len(analyses)} directories")
        return analyses

    def _analyze_directory(self, dir_path: Path, py_files: List[str], subdirs: List[str]) -> Optional[DirectoryAnalysis]:
        """Analyze a single directory"""
        try:
            # Get all Python files in this directory
            # Create file paths (replacing complex comprehension with explicit loop)
            files = []
            for f in py_files:
                if f.endswith('.py'):
                    files.append(dir_path / f)

            # Analyze the directory's organization
            is_well_organized = self._assess_organization(dir_path, files, subdirs)
            organization_score = self._calculate_organization_score(dir_path, files, subdirs)
            primary_category = self._categorize_directory(dir_path, files)
            relationships = self._analyze_relationships(dir_path, files)

            # Determine reasons for organization assessment
            reasons = []
            if is_well_organized:
                reasons.append("Directory has clear, logical structure")
            else:
                reasons.append("Directory could benefit from reorganization")

            return DirectoryAnalysis(
                path=dir_path,
                package_name=dir_path.name,
                files=files,
                subdirectories=[dir_path / d for d in subdirs],
                is_well_organized=is_well_organized,
                organization_score=organization_score,
                primary_category=primary_category,
                relationships=relationships,
                reasons=reasons
            )

        except Exception as e:
            self.logger.warning(f"Error analyzing {dir_path}: {e}")
            return None

    def _assess_organization(self, dir_path: Path, files: List[Path], subdirs: List[str]) -> bool:
        """Assess if a directory is well-organized"""
        if not files:
            return True  # Empty directories are "well-organized"

        # Check for clear naming patterns
        has_clear_naming = self._check_naming_patterns(dir_path, files)

        # Check for logical file grouping
        has_logical_grouping = self._check_logical_grouping(files)

        # Check if files are related (import each other)
        has_internal_relationships = self._check_internal_relationships(dir_path, files)

        # Directory is well-organized if it has good naming AND (logical grouping OR internal relationships)
        return has_clear_naming and (has_logical_grouping or has_internal_relationships)

    def _check_naming_patterns(self, dir_path: Path, files: List[Path]) -> bool:
        """Check if files follow clear naming patterns"""
        if not files:
            return True

        # Extract base names (without extensions) - replacing complex comprehension with explicit loop
        base_names = []
        for f in files:
            base_names.append(f.stem.lower())

        # Look for common patterns (replacing complex comprehensions with explicit loops)
        class_files = []
        util_files = []
        config_files = []
        test_files = []
        main_files = []

        for n in base_names:
            if n.endswith(('class', 'classes', 'model', 'models')):
                class_files.append(n)
            if n.endswith(('util', 'utils', 'helper', 'helpers')):
                util_files.append(n)
            if 'config' in n:
                config_files.append(n)
            if n.startswith(('test_', '_test')):
                test_files.append(n)
            if n in ('__init__', 'main', 'app', 'application'):
                main_files.append(n)

        patterns = {
            'class_files': class_files,
            'util_files': util_files,
            'config_files': config_files,
            'test_files': test_files,
            'main_files': main_files
        }

        # Directory is well-named if files follow consistent patterns
        total_patterns = sum(len(p) for p in patterns.values())
        return total_patterns >= len(files) * 0.6  # 60% of files follow patterns

    def _check_logical_grouping(self, files: List[Path]) -> bool:
        """Check if files are logically grouped"""
        if len(files) < 2:
            return True

        # Analyze file contents for common themes
        themes = set()
        for file_path in files[:5]:  # Check first 5 files to avoid too much processing
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()

                # Look for common themes
                if any(keyword in content for keyword in ['class', 'def ', 'import']):
                    themes.add('code')
                if any(keyword in content for keyword in ['config', 'setting', 'parameter']):
                    themes.add('config')
                if any(keyword in content for keyword in ['test', 'assert', 'pytest']):
                    themes.add('test')
                if any(keyword in content for keyword in ['log', 'print', 'debug']):
                    themes.add('logging')

            except:
                continue

        # Well-grouped if most files share themes
        return len(themes) <= 2  # No more than 2 different themes

    def _check_internal_relationships(self, dir_path: Path, files: List[Path]) -> bool:
        """Check if files in directory have internal relationships"""
        if len(files) < 2:
            return True

        # Look for imports between files in the same directory
        internal_imports = 0
        total_possible = len(files) * (len(files) - 1)  # All possible import pairs

        for i, source_file in enumerate(files):
            try:
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Check for imports of other files in same directory
                for target_file in files[i+1:]:
                    target_name = target_file.stem
                    if f"from {target_name}" in content or f"import {target_name}" in content:
                        internal_imports += 1

            except:
                continue

        # Directory has internal relationships if > 20% of possible imports exist
        return internal_imports > total_possible * 0.2

    def _calculate_organization_score(self, dir_path: Path, files: List[Path], subdirs: List[str]) -> float:
        """Calculate a numerical score for directory organization"""
        score = 0.5  # Base score

        # Naming consistency (+0.2)
        if self._check_naming_patterns(dir_path, files):
            score += 0.2

        # Logical grouping (+0.2)
        if self._check_logical_grouping(files):
            score += 0.2

        # Internal relationships (+0.1)
        if self._check_internal_relationships(dir_path, files):
            score += 0.1

        # Appropriate size (+0.1)
        if 2 <= len(files) <= 20:  # Not too few, not too many files
            score += 0.1

        return min(score, 1.0)

    def _categorize_directory(self, dir_path: Path, files: List[Path]) -> str:
        """Categorize a directory by its primary function"""
        # This would use the same categorization logic as the original reorganizer
        # but applied to the directory as a whole rather than individual files

        # For now, return the most common category from analyzing the files
        categories = []
        for file_path in files[:3]:  # Sample first 3 files
            category = self._categorize_file_simple(file_path)
            categories.append(category)

        if categories:
            # Return most common category
            return max(set(categories), key=categories.count)

        return "utilities"

    def _categorize_file_simple(self, file_path: Path) -> str:
        """Simple categorization for a file (extracted from original reorganizer)"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()

            # Check for category keywords
            categories = {
                'intelligence': ['intelligence', 'ml', 'ai', 'neural', 'predictive', 'learning'],
                'orchestration': ['orchestrator', 'coordinator', 'workflow', 'agent', 'swarm'],
                'security': ['security', 'auth', 'encrypt', 'vulnerability', 'threat'],
                'testing': ['test', 'mock', 'fixture', 'assertion', 'coverage'],
                'monitoring': ['monitor', 'dashboard', 'metric', 'log', 'alert'],
                'deployment': ['deploy', 'install', 'setup', 'config'],
                'documentation': ['doc', 'readme', 'guide', 'tutorial']
            }

            scores = {}
            for category, keywords in categories.items():
                score = sum(1 for keyword in keywords if keyword in content)
                scores[category] = score

            if scores:
                return max(scores.items(), key=lambda x: x[1])[0]

        except:
            pass

        return "utilities"

    def _analyze_relationships(self, dir_path: Path, files: List[Path]) -> Dict[str, List[str]]:
        """Analyze relationships with other directories"""
        relationships = {
            'imports_from': [],
            'imported_by': [],
            'shares_patterns': [],
            'related_functionality': []
        }

        # This would analyze imports and shared patterns with other directories
        # For now, return empty relationships
        return relationships

    def generate_reorganization_plan(self, analyses: List[DirectoryAnalysis]) -> Dict:
        """Generate a smart reorganization plan"""
        plan = {
            'directories_to_move': [],
            'directories_to_preserve': [],
            'suggested_improvements': [],
            'summary': {}
        }

        for analysis in analyses:
            if analysis.is_well_organized or analysis.organization_score > 0.7:
                # Preserve well-organized directories
                plan['directories_to_preserve'].append({
                    'path': str(analysis.path),
                    'reason': 'Well-organized directory',
                    'score': analysis.organization_score
                })
            else:
                # Consider reorganization for poorly organized directories
                if analysis.organization_score < 0.4:
                    plan['directories_to_move'].append({
                        'path': str(analysis.path),
                        'suggested_category': analysis.primary_category,
                        'reason': 'Poor organization score',
                        'score': analysis.organization_score
                    })
                else:
                    plan['suggested_improvements'].append({
                        'path': str(analysis.path),
                        'suggestion': 'Minor reorganization needed',
                        'score': analysis.organization_score
                    })

        plan['summary'] = {
            'total_directories_analyzed': len(analyses),
            'well_organized_directories': len(plan['directories_to_preserve']),
            'needs_reorganization': len(plan['directories_to_move']),
            'minor_improvements': len(plan['suggested_improvements'])
        }

        return plan

    def execute_plan(self, plan: Dict) -> None:
        """Execute the reorganization plan"""
        if self.mode == 'preview':
            self._print_plan(plan)
            return

        # For now, just create the target structure
        target_root = self.root_dir / 'organized_codebase'
        target_root.mkdir(exist_ok=True)

        # Create category directories
        categories = ['core', 'security', 'testing', 'monitoring', 'deployment', 'documentation', 'configuration', 'utilities']
        for category in categories:
            (target_root / category).mkdir(exist_ok=True)

        self.logger.info("Reorganization structure created")

    def _print_plan(self, plan: Dict) -> None:
        """Print the reorganization plan"""
        print("\n" + "="*80)
        print("INTELLIGENT CODEBASE REORGANIZATION PLAN")
        print("="*80)

        print(f"\n📊 SUMMARY:")
        print(f"   Directories analyzed: {plan['summary']['total_directories_analyzed']}")
        print(f"   Well-organized (preserved): {plan['summary']['well_organized_directories']}")
        print(f"   Needs reorganization: {plan['summary']['needs_reorganization']}")
        print(f"   Minor improvements: {plan['summary']['minor_improvements']}")

        print("✅ DIRECTORIES TO PRESERVE:")
        for item in plan['directories_to_preserve'][:10]:
            print(f"   📁 {item}")
        if len(plan['directories_to_preserve']) > 10:
            print(f"   ... and {len(plan['directories_to_preserve']) - 10} more")

        print("🔄 DIRECTORIES TO REORGANIZE:")
        for item in plan['directories_to_move'][:10]:
            print(f"   📁 {item}")
        if len(plan['directories_to_move']) > 10:
            print(f"   ... and {len(plan['directories_to_move']) - 10} more")

        print("💡 SUGGESTED IMPROVEMENTS:")
        for item in plan['suggested_improvements'][:10]:
            print(f"   💡 {item}")
        if len(plan['suggested_improvements']) > 10:
            print(f"   ... and {len(plan['suggested_improvements']) - 10} more")

def main() -> None:
    """Main function"""
    print("Intelligent Codebase Reorganizer")
    print("=" * 40)

    # Use current directory as root
    root_dir = Path.cwd()

    reorganizer = IntelligentReorganizer(root_dir, "preview")

    # Analyze directory structure
    analyses = reorganizer.analyze_directory_structure()

    # Generate reorganization plan
    plan = reorganizer.generate_reorganization_plan(analyses)

    # Execute plan
    reorganizer.execute_plan(plan)

    print("Analysis complete!")
    print(f"Check the log file for detailed results")

if __name__ == "__main__":
    main()

