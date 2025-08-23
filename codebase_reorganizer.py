#!/usr/bin/env python3
"""
Codebase Reorganization Script
=============================

Analyzes and reorganizes the TestMaster codebase into a clean, logical structure
similar to the production package organization.

Features:
- Non-destructive reorganization (uses symlinks by default)
- Intelligent categorization based on file analysis
- Import statement updates
- Backup and rollback capabilities
- Multiple execution modes (preview, interactive, automatic)

Usage:
    python codebase_reorganizer.py --preview    # See what would be done
    python codebase_reorganizer.py --interactive # Ask before each change
    python codebase_reorganizer.py --automatic  # Full automation (for OpenDevin)
    python codebase_reorganizer.py --symlinks   # Use symlinks (safest)
    python codebase_reorganizer.py --move       # Actually move files

Author: Codebase Reorganization System
"""

import os
import sys
import json
import shutil
import re
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    path: Path
    category: str
    confidence: float
    reasons: List[str]
    imports: List[str]
    classes: List[str]
    functions: List[str]
    keywords: Set[str]

class CodebaseReorganizer:
    def __init__(self, root_dir: Path, mode: str = "preview"):
        self.root_dir = root_dir
        self.mode = mode
        self.backup_dir = None

        # Setup logging
        self.setup_logging()

        # Define target structure (same as production package)
        self.target_structure = {
            'core': {
                'intelligence': [],
                'orchestration': [],
                'security': [],
                'foundation': [],
                'services': []
            },
            'security': [],
            'testing': [],
            'monitoring': [],
            'deployment': [],
            'documentation': [],
            'configuration': [],
            'utilities': []
        }

        # Categorization rules
        self.categorization_rules = self._build_categorization_rules()

        # Files to exclude from reorganization
        self.exclude_patterns = {
            '**/node_modules/**',
            '**/.*',
            '**/testmaster_sessions/**',
            '**/archives/**',
            'PRODUCTION_PACKAGES/**',
            'codebase_reorganizer.py'
        }

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.root_dir / f"reorganization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _build_categorization_rules(self) -> Dict:
        """Build comprehensive categorization rules"""
        return {
            'core/intelligence': {
                'keywords': {
                    'intelligence', 'ml', 'ai', 'neural', 'predictive', 'learning',
                    'classifier', 'regression', 'clustering', 'nlp', 'llm', 'gpt',
                    'transformer', 'embedding', 'vector', 'semantic', 'cognitive'
                },
                'path_patterns': [
                    r'.*intelligence.*', r'.*ml.*', r'.*ai.*', r'.*neural.*',
                    r'.*predictive.*', r'.*semantic.*', r'.*cognitive.*'
                ],
                'class_patterns': [
                    r'.*Intelligence.*', r'.*ML.*', r'.*AI.*', r'.*Neural.*',
                    r'.*Predictor.*', r'.*Classifier.*', r'.*Learner.*'
                ]
            },
            'core/orchestration': {
                'keywords': {
                    'orchestrator', 'orchestration', 'coordinator', 'coordination',
                    'workflow', 'pipeline', 'scheduler', 'dispatcher', 'agent',
                    'swarm', 'coordination', 'messaging', 'queue', 'task',
                    'executor', 'conductor', 'director'
                },
                'path_patterns': [
                    r'.*orchestrat.*', r'.*coordinat.*', r'.*workflow.*',
                    r'.*agent.*', r'.*swarm.*', r'.*messaging.*', r'.*queue.*'
                ],
                'class_patterns': [
                    r'.*Orchestrator.*', r'.*Coordinator.*', r'.*Agent.*',
                    r'.*Scheduler.*', r'.*Workflow.*', r'.*Executor.*'
                ]
            },
            'core/security': {
                'keywords': {
                    'security', 'auth', 'authentication', 'authorization', 'encrypt',
                    'decrypt', 'hash', 'password', 'token', 'jwt', 'oauth',
                    'vulnerability', 'scan', 'audit', 'compliance', 'threat',
                    'firewall', 'access', 'permission', 'credential', 'secret'
                },
                'path_patterns': [
                    r'.*security.*', r'.*auth.*', r'.*encrypt.*', r'.*audit.*',
                    r'.*threat.*', r'.*compliance.*', r'.*credential.*'
                ],
                'class_patterns': [
                    r'.*Security.*', r'.*Auth.*', r'.*Encrypt.*', r'.*Audit.*',
                    r'.*Compliance.*', r'.*Threat.*', r'.*Scanner.*'
                ]
            },
            'core/foundation': {
                'keywords': {
                    'base', 'abstract', 'interface', 'foundation', 'core',
                    'framework', 'abstraction', 'protocol', 'contract',
                    'utility', 'helper', 'common', 'shared', 'library'
                },
                'path_patterns': [
                    r'.*foundation.*', r'.*base.*', r'.*abstract.*',
                    r'.*interface.*', r'.*core.*', r'.*framework.*'
                ],
                'class_patterns': [
                    r'.*Base.*', r'.*Abstract.*', r'.*Interface.*',
                    r'.*Foundation.*', r'.*Core.*', r'.*Helper.*'
                ]
            },
            'security': {
                'keywords': {
                    'patch', 'fix', 'vulnerability', 'exploit', 'injection',
                    'xss', 'csrf', 'sql', 'path', 'traversal', 'validation',
                    'sanitize', 'escape', 'filter', 'guard', 'protection'
                },
                'path_patterns': [
                    r'.*patch.*', r'.*fix.*', r'.*security.*', r'.*vulnerab.*'
                ]
            },
            'testing': {
                'keywords': {
                    'test', 'spec', 'mock', 'stub', 'fixture', 'assertion',
                    'coverage', 'pytest', 'unittest', 'nose', 'behave',
                    'cucumber', 'selenium', 'automation', 'validation'
                },
                'path_patterns': [
                    r'.*test.*', r'.*spec.*', r'.*mock.*', r'.*fixture.*'
                ]
            },
            'monitoring': {
                'keywords': {
                    'monitor', 'dashboard', 'metric', 'log', 'alert',
                    'notification', 'observability', 'telemetry', 'trace',
                    'performance', 'health', 'status', 'report', 'analytics'
                },
                'path_patterns': [
                    r'.*monitor.*', r'.*dashboard.*', r'.*metric.*',
                    r'.*log.*', r'.*alert.*', r'.*telemetry.*'
                ]
            },
            'deployment': {
                'keywords': {
                    'deploy', 'install', 'setup', 'config', 'environment',
                    'docker', 'kubernetes', 'aws', 'azure', 'gcp',
                    'server', 'production', 'staging', 'devops', 'ci', 'cd'
                },
                'path_patterns': [
                    r'.*deploy.*', r'.*install.*', r'.*setup.*',
                    r'.*docker.*', r'.*config.*', r'.*environment.*'
                ]
            },
            'documentation': {
                'keywords': {
                    'doc', 'readme', 'guide', 'tutorial', 'example',
                    'documentation', 'manual', 'reference', 'api'
                },
                'path_patterns': [
                    r'.*readme.*', r'.*doc.*', r'.*guide.*', r'.*tutorial.*'
                ]
            },
            'configuration': {
                'keywords': {
                    'config', 'setting', 'parameter', 'option', 'preference',
                    'env', 'environment', 'variable', 'constant', 'default'
                },
                'path_patterns': [
                    r'.*config.*', r'.*setting.*', r'.*env.*', r'.*constant.*'
                ]
            }
        }

    def should_exclude(self, path: Path) -> bool:
        """Check if a file should be excluded from reorganization"""
        path_str = str(path.relative_to(self.root_dir))

        for pattern in self.exclude_patterns:
            if path.match(pattern):
                return True

        # Exclude non-Python files (for now)
        if path.suffix != '.py':
            return True

        return False

    def analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """Analyze a single file to determine its category"""
        if self.should_exclude(file_path):
            return None

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract basic info
            imports = self._extract_imports(content)
            classes = self._extract_classes(content)
            functions = self._extract_functions(content)
            keywords = self._extract_keywords(content)

            # Analyze content for categorization
            category, confidence, reasons = self._categorize_file(
                file_path, content, imports, classes, functions, keywords
            )

            return FileAnalysis(
                path=file_path,
                category=category,
                confidence=confidence,
                reasons=reasons,
                imports=imports,
                classes=classes,
                functions=functions,
                keywords=keywords
            )

        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path}: {e}")
            return None

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from file"""
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            # Fallback: regex-based extraction
            import_pattern = r'^(?:from|import)\s+([^\s;]+)'
            imports = re.findall(import_pattern, content, re.MULTILINE)

        return imports

    def _extract_classes(self, content: str) -> List[str]:
        """Extract class names from file"""
        classes = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except:
            # Fallback: regex-based extraction
            class_pattern = r'^class\s+(\w+)'
            classes = re.findall(class_pattern, content, re.MULTILINE)

        return classes

    def _extract_functions(self, content: str) -> List[str]:
        """Extract function names from file"""
        functions = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
        except:
            # Fallback: regex-based extraction
            func_pattern = r'^def\s+(\w+)'
            functions = re.findall(func_pattern, content, re.MULTILINE)

        return functions

    def _extract_keywords(self, content: str) -> Set[str]:
        """Extract meaningful keywords from file"""
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', content.lower())

        # Filter out common programming keywords and keep meaningful ones
        stop_words = {
            'def', 'class', 'if', 'else', 'for', 'while', 'try', 'except',
            'with', 'as', 'in', 'is', 'not', 'and', 'or', 'import', 'from',
            'return', 'self', 'true', 'false', 'none', 'int', 'str', 'list',
            'dict', 'set', 'tuple', 'len', 'range', 'print', 'open', 'file'
        }

        keywords = set(word for word in words if len(word) > 3 and word not in stop_words)
        return keywords

    def _categorize_file(self, path: Path, content: str,
                        imports: List[str], classes: List[str],
                        functions: List[str], keywords: Set[str]) -> Tuple[str, float, List[str]]:
        """Categorize a file based on its content and structure"""

        path_str = str(path.relative_to(self.root_dir)).lower()
        scores = {}

        for category, rules in self.categorization_rules.items():
            score = 0
            reasons = []

            # Path-based scoring
            for pattern in rules.get('path_patterns', []):
                if re.search(pattern, path_str, re.IGNORECASE):
                    score += 0.3
                    reasons.append(f"Path matches pattern: {pattern}")

            # Keyword-based scoring
            rule_keywords = rules.get('keywords', set())
            matching_keywords = keywords.intersection(rule_keywords)
            if matching_keywords:
                keyword_score = len(matching_keywords) / len(keywords) if keywords else 0
                score += keyword_score * 0.4
                reasons.append(f"Found keywords: {', '.join(list(matching_keywords)[:5])}")

            # Class name scoring
            for pattern in rules.get('class_patterns', []):
                for class_name in classes:
                    if re.search(pattern, class_name, re.IGNORECASE):
                        score += 0.2
                        reasons.append(f"Class matches pattern: {class_name}")

            # Import-based scoring
            for imp in imports:
                imp_lower = imp.lower()
                if any(keyword in imp_lower for keyword in rule_keywords):
                    score += 0.1
                    reasons.append(f"Import suggests category: {imp}")

            scores[category] = (score, reasons)

        # Find best category
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1][0])
            return best_category[0], min(best_category[1][0], 1.0), best_category[1][1]

        # Default to utilities if no good match
        return 'utilities', 0.1, ['Default categorization']

    def analyze_codebase(self) -> List[FileAnalysis]:
        """Analyze all files in the codebase"""
        self.logger.info("Starting codebase analysis...")

        analyses = []

        for root, dirs, files in os.walk(self.root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                (Path(root) / d).match(pattern) for pattern in self.exclude_patterns
            )]

            for file in files:
                file_path = Path(root) / file
                if not self.should_exclude(file_path):
                    analysis = self.analyze_file(file_path)
                    if analysis:
                        analyses.append(analysis)
                        if len(analyses) % 100 == 0:
                            self.logger.info(f"Analyzed {len(analyses)} files...")

        self.logger.info(f"Completed analysis of {len(analyses)} files")
        return analyses

    def generate_reorganization_plan(self, analyses: List[FileAnalysis]) -> Dict:
        """Generate a reorganization plan based on file analyses"""
        plan = {
            'moves': [],
            'symlinks': [],
            'imports_to_update': [],
            'summary': {}
        }

        for analysis in analyses:
            target_path = self._get_target_path(analysis)

            if analysis.path != target_path:
                if self.mode == 'symlinks':
                    plan['symlinks'].append({
                        'source': str(analysis.path),
                        'target': str(target_path),
                        'category': analysis.category,
                        'confidence': analysis.confidence
                    })
                else:
                    plan['moves'].append({
                        'source': str(analysis.path),
                        'target': str(target_path),
                        'category': analysis.category,
                        'confidence': analysis.confidence
                    })

                # Check for imports that need updating
                imports_to_update = self._find_imports_to_update(analysis.path, target_path)
                plan['imports_to_update'].extend(imports_to_update)

        # Generate summary
        plan['summary'] = self._generate_summary(plan)

        return plan

    def _get_target_path(self, analysis: FileAnalysis) -> Path:
        """Determine the target path for a file"""
        # Create target directory structure
        category_parts = analysis.category.split('/')
        target_dir = self.root_dir / 'organized_codebase'
        for part in category_parts:
            target_dir = target_dir / part

        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create target file path
        return target_dir / analysis.path.name

    def _find_imports_to_update(self, source_path: Path, target_path: Path) -> List[Dict]:
        """Find files that import from the moved file"""
        imports_to_update = []

        # This is a simplified version - a full implementation would need
        # to search all Python files for imports of this module
        module_name = source_path.stem
        new_module_path = target_path.relative_to(self.root_dir)

        # Find potential importing files (files that might import this module)
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py') and not self.should_exclude(Path(root) / file):
                    try:
                        with open(Path(root) / file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if module_name in content:
                                imports_to_update.append({
                                    'file': str(Path(root) / file),
                                    'old_import': module_name,
                                    'new_import': str(new_module_path.with_suffix('')).replace('\\', '.')
                                })
                    except:
                        pass

        return imports_to_update

    def _generate_summary(self, plan: Dict) -> Dict:
        """Generate a summary of the reorganization plan"""
        summary = {
            'total_files_to_move': len(plan['moves']),
            'total_symlinks_to_create': len(plan['symlinks']),
            'imports_to_update': len(plan['imports_to_update']),
            'categories': {}
        }

        # Count by category
        all_operations = plan['moves'] + plan['symlinks']
        for operation in all_operations:
            category = operation['category']
            if category not in summary['categories']:
                summary['categories'][category] = 0
            summary['categories'][category] += 1

        return summary

    def execute_plan(self, plan: Dict):
        """Execute the reorganization plan"""
        if self.mode == 'preview':
            self._print_plan(plan)
            return

        # Create backup
        self._create_backup()

        if self.mode in ['automatic', 'move']:
            self._execute_moves(plan['moves'])
            self._update_imports(plan['imports_to_update'])

        if self.mode == 'symlinks':
            self._create_symlinks(plan['symlinks'])

        self.logger.info("Reorganization completed")

    def _print_plan(self, plan: Dict):
        """Print the reorganization plan"""
        print("\n" + "="*80)
        print("CODEBASE REORGANIZATION PLAN")
        print("="*80)

        print(f"\n📊 SUMMARY:")
        print(f"   Files to reorganize: {plan['summary']['total_files_to_move']}")
        print(f"   Symlinks to create: {plan['summary']['total_symlinks_to_create']}")
        print(f"   Imports to update: {plan['summary']['imports_to_update']}")

        print(f"\n📁 CATEGORIES:")
        for category, count in plan['summary']['categories'].items():
            print(f"   {category}: {count} files")

        print(f"\n📋 DETAILED PLAN:")
        all_operations = plan['moves'] + plan['symlinks']
        for operation in all_operations[:20]:  # Show first 20
            action = "MOVE" if operation in plan['moves'] else "SYMLINK"
            print(f"   {action}: {operation['source']} -> {operation['target']}")
            print(f"   Category: {operation['category']} (confidence: {operation['confidence']:.2f})")

        if len(all_operations) > 20:
            print(f"   ... and {len(all_operations) - 20} more operations")

        print(f"\n🔄 IMPORT UPDATES:")
        for update in plan['imports_to_update'][:10]:
            print(f"   {update['file']}: {update['old_import']} -> {update['new_import']}")

        if len(plan['imports_to_update']) > 10:
            print(f"   ... and {len(plan['imports_to_update']) - 10} more updates")

    def _create_backup(self):
        """Create a backup of the current codebase"""
        self.backup_dir = self.root_dir / f"codebase_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(exist_ok=True)

        self.logger.info(f"Creating backup at {self.backup_dir}")

        # Copy all Python files to backup
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py'):
                    src = Path(root) / file
                    if not self.should_exclude(src):
                        rel_path = src.relative_to(self.root_dir)
                        dst = self.backup_dir / rel_path
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)

    def _execute_moves(self, moves: List[Dict]):
        """Execute file moves"""
        for move in moves:
            source = Path(move['source'])
            target = Path(move['target'])

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, target)
                self.logger.info(f"Moved: {source} -> {target}")
            except Exception as e:
                self.logger.error(f"Failed to move {source} -> {target}: {e}")

    def _create_symlinks(self, symlinks: List[Dict]):
        """Create symlinks"""
        for symlink in symlinks:
            source = Path(symlink['source'])
            target = Path(symlink['target'])

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Remove existing file if it exists
                if target.exists():
                    target.unlink()
                target.symlink_to(source)
                self.logger.info(f"Symlinked: {source} -> {target}")
            except Exception as e:
                self.logger.error(f"Failed to symlink {source} -> {target}: {e}")

    def _update_imports(self, imports_to_update: List[Dict]):
        """Update import statements in files"""
        for update in imports_to_update:
            file_path = Path(update['file'])
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Simple import replacement (could be more sophisticated)
                old_import = update['old_import']
                new_import = update['new_import']

                # Replace import statements
                content = re.sub(
                    rf'\b{re.escape(old_import)}\b',
                    new_import,
                    content
                )

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.logger.info(f"Updated imports in {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to update imports in {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Codebase Reorganization Tool')
    parser.add_argument('--mode', choices=['preview', 'interactive', 'automatic', 'symlinks', 'move'],
                       default='preview', help='Execution mode')
    parser.add_argument('--root', type=str, default='.',
                       help='Root directory of the codebase')

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()

    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist")
        sys.exit(1)

    print(f"Codebase Reorganization Tool")
    print(f"Root Directory: {root_dir}")
    print(f"Mode: {args.mode}")
    print("-" * 50)

    reorganizer = CodebaseReorganizer(root_dir, args.mode)

    # Analyze codebase
    analyses = reorganizer.analyze_codebase()

    # Generate reorganization plan
    plan = reorganizer.generate_reorganization_plan(analyses)

    # Execute plan
    reorganizer.execute_plan(plan)

    print("\nReorganization complete!")
    print(f"Check the log file for details: reorganization_*.log")

if __name__ == "__main__":
    main()
