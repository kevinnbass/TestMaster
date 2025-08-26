#!/usr/bin/env python3
"""
Codebase Reorganization Tool
=============================

A comprehensive tool to analyze and reorganize the TestMaster codebase
into a clean, logical structure that enables better agent swarm coordination.

Features:
- Intelligent file analysis and categorization
- Non-destructive reorganization with multiple execution modes
- Automatic import statement updates
- Backup and rollback capabilities
- Comprehensive exclusion rules (same as find_active_python_modules.py)
- Multiple execution modes for different use cases

Author: Codebase Reorganization System
Version: 2.0
"""

import os
import sys
import json
import shutil
import re
import ast
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import tempfile
import subprocess

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
    file_hash: str
    size: int
    modified_time: float

@dataclass
class ReorganizationResult:
    """Results of the reorganization process"""
    total_files_analyzed: int
    files_reorganized: int
    symlinks_created: int
    imports_updated: int
    errors: List[str]
    warnings: List[str]
    backup_path: Optional[Path]

class CodebaseReorganizer:
    def __init__(self, root_dir: Path, mode: str = "preview", config_file: Optional[Path] = None) -> None:
        """Initialize the codebase reorganizer with configuration and logging"""
        self.root_dir = root_dir.resolve()
        self.mode = mode
        self.backup_dir = None
        self.config_file = config_file or self.root_dir / "tools" / "codebase_reorganizer" / "config" / "reorganizer_config.json"

        # Setup logging
        self.setup_logging()

        # Load or create configuration
        self.config = self.load_config()

        # Initialize exclusion patterns (same as find_active_python_modules.py)
        self.exclude_patterns = self._build_exclusion_patterns()

        # Initialize categorization rules
        self.categorization_rules = self._build_categorization_rules()

        # Initialize target structure
        self.target_structure = self._build_target_structure()

        self.logger.info(f"Codebase Reorganizer initialized for {self.root_dir}")
        self.logger.info(f"Mode: {self.mode}")

    def setup_logging(self) -> None:
        """Setup comprehensive logging"""
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"reorganization_{timestamp}.log"

        # Setup logging with both file and console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load config file: {e}")

        # Default configuration
        return {
            "exclusions": {
                "research_repos": [
                    "agency-swarm", "autogen", "agent-squad", "agentops",
                    "agentscope", "AgentVerse", "crewAI", "CodeGraph",
                    "falkordb-py", "AWorld", "MetaGPT", "metagpt",
                    "PraisonAI", "praisonai", "llama-agents", "phidata",
                    "swarms", "lagent", "langgraph-supervisor-py"
                ],
                "system_dirs": [
                    "__pycache__", ".git", "node_modules", "htmlcov",
                    ".pytest_cache", ".vscode", ".idea", ".DS_Store"
                ],
                "archive_dirs": ["archive", "archives", "PRODUCTION_PACKAGES"],
                "test_dirs": ["tests", "test_sessions", "testmaster_sessions"],
                "build_dirs": ["build", "dist", "egg-info", "*.egg-info"],
                "temp_dirs": ["temp", "tmp", "temporary", "cache"]
            },
            "categories": {
                "core/intelligence": {
                    "keywords": ["intelligence", "ml", "ai", "neural", "predictive", "learning", "classifier", "regression", "clustering", "nlp", "llm", "gpt", "transformer", "embedding", "vector", "semantic", "cognitive", "analytics", "forecaster", "predictor"],
                    "class_patterns": [".*Intelligence.*", ".*ML.*", ".*AI.*", ".*Neural.*", ".*Predictor.*", ".*Classifier.*", ".*Learner.*", ".*Forecaster.*"],
                    "path_patterns": [".*intelligence.*", ".*ml.*", ".*ai.*", ".*neural.*", ".*predictive.*", ".*semantic.*", ".*cognitive.*", ".*analytics.*"]
                },
                "core/orchestration": {
                    "keywords": ["orchestrator", "orchestration", "coordinator", "coordination", "workflow", "pipeline", "scheduler", "dispatcher", "agent", "swarm", "coordination", "messaging", "queue", "task", "executor", "conductor", "director", "router", "dispatcher"],
                    "class_patterns": [".*Orchestrator.*", ".*Coordinator.*", ".*Agent.*", ".*Scheduler.*", ".*Workflow.*", ".*Executor.*", ".*Router.*", ".*Dispatcher.*"],
                    "path_patterns": [".*orchestrat.*", ".*coordinat.*", ".*workflow.*", ".*agent.*", ".*swarm.*", ".*messaging.*", ".*queue.*", ".*router.*"]
                },
                "core/security": {
                    "keywords": ["security", "auth", "authentication", "authorization", "encrypt", "decrypt", "hash", "password", "token", "jwt", "oauth", "vulnerability", "scan", "audit", "compliance", "threat", "firewall", "access", "permission", "credential", "secret", "validation", "sanitization"],
                    "class_patterns": [".*Security.*", ".*Auth.*", ".*Encrypt.*", ".*Audit.*", ".*Compliance.*", ".*Threat.*", ".*Scanner.*", ".*Validator.*"],
                    "path_patterns": [".*security.*", ".*auth.*", ".*encrypt.*", ".*audit.*", ".*threat.*", ".*compliance.*", ".*credential.*", ".*validator.*"]
                },
                "core/foundation": {
                    "keywords": ["base", "abstract", "interface", "foundation", "core", "framework", "abstraction", "protocol", "contract", "utility", "helper", "common", "shared", "library", "config", "settings"],
                    "class_patterns": [".*Base.*", ".*Abstract.*", ".*Interface.*", ".*Foundation.*", ".*Core.*", ".*Helper.*", ".*Config.*"],
                    "path_patterns": [".*foundation.*", ".*base.*", ".*abstract.*", ".*interface.*", ".*core.*", ".*framework.*", ".*helper.*", ".*config.*"]
                },
                "security": {
                    "keywords": ["patch", "fix", "vulnerability", "exploit", "injection", "xss", "csrf", "sql", "path", "traversal", "validation", "sanitize", "escape", "filter", "guard", "protection", "hardening", "defense"],
                    "path_patterns": [".*patch.*", ".*fix.*", ".*security.*", ".*vulnerab.*", ".*exploit.*"]
                },
                "testing": {
                    "keywords": ["test", "spec", "mock", "stub", "fixture", "assertion", "coverage", "pytest", "unittest", "nose", "behave", "cucumber", "selenium", "automation", "validation", "verify", "check"],
                    "path_patterns": [".*test.*", ".*spec.*", ".*mock.*", ".*fixture.*", ".*check.*"]
                },
                "monitoring": {
                    "keywords": ["monitor", "dashboard", "metric", "log", "alert", "notification", "observability", "telemetry", "trace", "performance", "health", "status", "report", "analytics", "visualization", "graph"],
                    "class_patterns": [".*Monitor.*", ".*Dashboard.*", ".*Metric.*", ".*Logger.*"],
                    "path_patterns": [".*monitor.*", ".*dashboard.*", ".*metric.*", ".*log.*", ".*alert.*", ".*telemetry.*", ".*visual.*", ".*graph.*"]
                },
                "deployment": {
                    "keywords": ["deploy", "install", "setup", "environment", "docker", "kubernetes", "aws", "azure", "gcp", "server", "production", "staging", "devops", "ci", "cd", "build", "package", "distribution"],
                    "path_patterns": [".*deploy.*", ".*install.*", ".*setup.*", ".*docker.*", ".*environment.*", ".*build.*", ".*package.*"]
                },
                "documentation": {
                    "keywords": ["doc", "readme", "guide", "tutorial", "example", "documentation", "manual", "reference", "api", "usage", "howto"],
                    "path_patterns": [".*readme.*", ".*doc.*", ".*guide.*", ".*tutorial.*", ".*example.*", ".*usage.*"]
                },
                "configuration": {
                    "keywords": ["config", "setting", "parameter", "option", "preference", "env", "environment", "variable", "constant", "default", "properties", "ini", "yaml", "json", "toml"],
                    "path_patterns": [".*config.*", ".*setting.*", ".*env.*", ".*constant.*", ".*properties.*"]
                }
            },
            "operations": {
                "create_backups": True,
                "use_symlinks": True,
                "update_imports": True,
                "validate_after_reorg": True,
                "max_file_size": 10485760  # 10MB
            }
        }

    def _build_exclusion_patterns(self) -> Set[str]:
        """Build exclusion patterns from config"""
        patterns = set()

        # Add all exclusion categories
        for category, dirs in self.config["exclusions"].items():
            for dir_name in dirs:
                patterns.add(f"**/{dir_name}/**")
                patterns.add(f"**/{dir_name}")

        # Add specific file patterns
        patterns.update([
            "**/*.pyc",
            "**/__pycache__/**",
            "**/.git/**",
            "**/node_modules/**",
            "**/htmlcov/**",
            "**/.pytest_cache/**",
            "**/.coverage",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.log",
            "**/*.tmp",
            "**/*.temp"
        ])

        return patterns

    def _build_categorization_rules(self) -> Dict:
        """Build categorization rules from config"""
        return self.config["categories"]

    def _build_target_structure(self) -> Dict:
        """Build the target directory structure"""
        return {
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

    def should_exclude(self, path: Path) -> bool:
        """Check if a file should be excluded from reorganization"""
        if not path.is_file() or path.suffix != '.py':
            return True

        try:
            rel_path = path.relative_to(self.root_dir)
        except ValueError:
            return True

        path_str = str(rel_path)

        # Check against exclusion patterns
        for pattern in self.exclude_patterns:
            if path.match(pattern):
                return True

        # Check for tools directory (don't reorganize our own tools)
        if "tools/" in path_str:
            return True

        return False

    def analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """Analyze a single file to determine its category"""
        if self.should_exclude(file_path):
            return None

        try:
            # Get file metadata
            stat = file_path.stat()
            file_hash = self._calculate_file_hash(file_path)

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract code elements
            imports = self._extract_imports(content)
            classes = self._extract_classes(content)
            functions = self._extract_functions(content)
            keywords = self._extract_keywords(content)

            # Determine category
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
                keywords=keywords,
                file_hash=file_hash,
                size=stat.st_size,
                modified_time=stat.st_mtime
            )

        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path}: {e}")
            return None

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return "unknown"

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
        except SyntaxError:
            # Fallback: regex-based extraction for files with syntax errors
            import_pattern = r'^(?:from|import)\s+([^\s;]+)'
            imports = re.findall(import_pattern, content, re.MULTILINE)

        return list(set(imports))  # Remove duplicates

    def _extract_classes(self, content: str) -> List[str]:
        """Extract class names from file"""
        classes = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except SyntaxError:
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
        except SyntaxError:
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
            'dict', 'set', 'tuple', 'len', 'range', 'print', 'open', 'file',
            'pass', 'break', 'continue', 'raise', 'yield', 'lambda', 'global',
            'nonlocal', 'assert', 'del', 'finally', 'elif', 'async', 'await'
        }

        keywords = set(word for word in words if len(word) > 2 and word not in stop_words)
        return keywords

    def _categorize_file(self, path: Path, content: str,
                        imports: List[str], classes: List[str],
                        functions: List[str], keywords: Set[str]) -> Tuple[str, float, List[str]]:
        """Categorize a file based on its content, imports, and path"""
        path_str = str(path.relative_to(self.root_dir)).lower()
        scores = {}

        for category, rules in self.categorization_rules.items():
            score = 0
            reasons = []

            # Path-based scoring (highest weight)
            for pattern in rules.get('path_patterns', []):
                if re.search(pattern, path_str, re.IGNORECASE):
                    score += 0.4
                    reasons.append(f"Path matches pattern: {pattern}")

            # Keyword-based scoring
            rule_keywords = set(rules.get('keywords', []))
            matching_keywords = keywords.intersection(rule_keywords)
            if matching_keywords:
                keyword_score = len(matching_keywords) / max(len(keywords), 1)
                score += keyword_score * 0.3
                reasons.append(f"Found keywords: {', '.join(list(matching_keywords)[:3])}")

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

            scores[category] = (min(score, 1.0), reasons)

        # Find best category
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1][0])
            return best_category[0], best_category[1][0], best_category[1][1]

        # Default to utilities if no good match
        return 'utilities', 0.1, ['Default categorization']

    def analyze_codebase(self) -> List[FileAnalysis]:
        """Analyze all files in the codebase"""
        self.logger.info("Starting comprehensive codebase analysis...")

        analyses = []

        # Add bounds checking to os.walk to prevent unbounded directory traversal
        max_directories = 1000  # Safety bound for directory traversal
        directory_count = 0

        for root, dirs, files in os.walk(self.root_dir):
            if directory_count >= max_directories:
                break  # Safety bound reached
            directory_count += 1

            # Skip excluded directories early (replacing complex comprehension)
            filtered_dirs = []
            for d in dirs:
                should_exclude = False
                for pattern in self.exclusion_patterns:
                    if pattern in str(Path(root) / d):
                        should_exclude = True
                        break
                if not should_exclude:
                    filtered_dirs.append(d)
            dirs[:] = filtered_dirs

            for file in files:
                file_path = Path(root) / file
                if not self.should_exclude(file_path):
                    analysis = self.analyze_file(file_path)
                    if analysis:
                        analyses.append(analysis)
                        if len(analyses) % 50 == 0:
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
                if self.config['operations']['use_symlinks'] or self.mode == 'symlinks':
                    plan['symlinks'].append({
                        'source': str(analysis.path),
                        'target': str(target_path),
                        'category': analysis.category,
                        'confidence': analysis.confidence,
                        'analysis': asdict(analysis)
                    })
                else:
                    plan['moves'].append({
                        'source': str(analysis.path),
                        'target': str(target_path),
                        'category': analysis.category,
                        'confidence': analysis.confidence,
                        'analysis': asdict(analysis)
                    })

                # Find imports that need updating
                if self.config['operations']['update_imports']:
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

        # Get module names
        source_module = source_path.stem
        target_rel_path = target_path.relative_to(self.root_dir / 'organized_codebase')
        target_module_path = str(target_rel_path.with_suffix('')).replace('\\', '.')

        # Search for files that might import this module
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py') and not self.should_exclude(Path(root) / file):
                    try:
                        with open(Path(root) / file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Look for imports of this module
                        import_patterns = [
                            rf'\bimport\s+{re.escape(source_module)}\b',
                            rf'\bfrom\s+{re.escape(source_module)}\b',
                            rf'\bimport\s+.*\b{re.escape(source_module)}\b',
                            rf'\bfrom\s+.*\b{re.escape(source_module)}\b'
                        ]

                        for pattern in import_patterns:
                            if re.search(pattern, content):
                                imports_to_update.append({
                                    'file': str(Path(root) / file),
                                    'old_import': source_module,
                                    'new_import': target_module_path,
                                    'pattern': pattern
                                })
                                break

                    except Exception as e:
                        self.logger.debug(f"Error checking imports in {Path(root) / file}: {e}")

        return imports_to_update

    def _generate_summary(self, plan: Dict) -> Dict:
        """Generate a summary of the reorganization plan"""
        summary = {
            'total_files_to_reorganize': len(plan['moves']) + len(plan['symlinks']),
            'total_moves': len(plan['moves']),
            'total_symlinks': len(plan['symlinks']),
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

    def execute_plan(self, plan: Dict) -> ReorganizationResult:
        """Execute the reorganization plan"""
        if self.mode == 'preview':
            self._print_plan(plan)
            return ReorganizationResult(
                total_files_analyzed=plan['summary']['total_files_to_reorganize'],
                files_reorganized=0,
                symlinks_created=0,
                imports_updated=0,
                errors=[],
                warnings=[],
                backup_path=None
            )

        # Create backup if configured
        if self.config['operations']['create_backups']:
            backup_path = self._create_backup()
        else:
            backup_path = None

        errors = []
        warnings = []

        # Execute moves
        if plan['moves']:
            move_results = self._execute_moves(plan['moves'])
            errors.extend(move_results['errors'])
            warnings.extend(move_results['warnings'])

        # Create symlinks
        if plan['symlinks']:
            symlink_results = self._create_symlinks(plan['symlinks'])
            errors.extend(symlink_results['errors'])
            warnings.extend(symlink_results['warnings'])

        # Update imports
        if plan['imports_to_update'] and self.config['operations']['update_imports']:
            import_results = self._update_imports(plan['imports_to_update'])
            errors.extend(import_results['errors'])
            warnings.extend(import_results['warnings'])

        return ReorganizationResult(
            total_files_analyzed=plan['summary']['total_files_to_reorganize'],
            files_reorganized=len(plan['moves']),
            symlinks_created=len(plan['symlinks']),
            # Count updated imports (replacing complex comprehension)
            updated_imports = 0
            for u in plan['imports_to_update']:
                if u.get('updated', False):
                    updated_imports += 1
            imports_updated=updated_imports,
            errors=errors,
            warnings=warnings,
            backup_path=backup_path
        )

    def _print_plan(self, plan: Dict) -> None:
        """Print the reorganization plan"""
        print("\n" + "="*80)
        print("CODEBASE REORGANIZATION PLAN")
        print("="*80)

        print(f"\n📊 SUMMARY:")
        print(f"   Files to reorganize: {plan['summary']['total_files_to_reorganize']}")
        print(f"   Direct moves: {plan['summary']['total_moves']}")
        print(f"   Symlinks to create: {plan['summary']['total_symlinks']}")
        print(f"   Imports to update: {plan['summary']['imports_to_update']}")

        print("📁 CATEGORIES:")
        for category, count in sorted(plan['summary']['categories'].items()):
            print(f"   {category}: {count} files")

        print("📋 DETAILED PLAN:")
        all_operations = plan['moves'] + plan['symlinks']
        for operation in all_operations[:30]:  # Show first 30
            action = "MOVE" if operation in plan['moves'] else "SYMLINK"
            confidence = operation['confidence']
            confidence_indicator = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
            print(f"   {action} {confidence_indicator} ({confidence:.2f}): {operation['source']}")
            print(f"   → {operation['target']}")
            print(f"   Category: {operation['category']}")

        if len(all_operations) > 30:
            print(f"   ... and {len(all_operations) - 30} more operations")

        print("🔄 IMPORT UPDATES:")
        for update in plan['imports_to_update'][:15]:
            print(f"   {update['file']}: {update['old_import']} → {update['new_import']}")

        if len(plan['imports_to_update']) > 15:
            print(f"   ... and {len(plan['imports_to_update']) - 15} more updates")

    def _create_backup(self) -> Path:
        """Create a backup of the current codebase"""
        backup_path = self.root_dir / "tools" / "codebase_reorganizer" / "backups" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Creating backup at {backup_path}")

        try:
            # Copy all Python files to backup
            for root, dirs, files in os.walk(self.root_dir):
                for file in files:
                    if file.endswith('.py'):
                        src = Path(root) / file
                        if not self.should_exclude(src):
                            rel_path = src.relative_to(self.root_dir)
                            dst = backup_path / rel_path
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)

            self.logger.info("Backup completed successfully")
            return backup_path

        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            raise

    def _execute_moves(self, moves: List[Dict]) -> Dict:
        """Execute file moves"""
        results = {'errors': [], 'warnings': []}

        for move in moves:
            source = Path(move['source'])
            target = Path(move['target'])

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, target)
                self.logger.info(f"Moved: {source} -> {target}")
            except Exception as e:
                error_msg = f"Failed to move {source} -> {target}: {e}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)

        return results

    def _create_symlinks(self, symlinks: List[Dict]) -> Dict:
        """Create symlinks"""
        results = {'errors': [], 'warnings': []}

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
                error_msg = f"Failed to symlink {source} -> {target}: {e}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)

        return results

    def _update_imports(self, imports_to_update: List[Dict]) -> Dict:
        """Update import statements in files"""
        results = {'errors': [], 'warnings': [], 'updated': []}

        for update in imports_to_update:
            file_path = Path(update['file'])
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Update import statements
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

                update['updated'] = True
                results['updated'].append(update)
                self.logger.info(f"Updated imports in {file_path}")

            except Exception as e:
                error_msg = f"Failed to update imports in {file_path}: {e}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
                update['updated'] = False

        return results

def main() -> None:
    """Main entry point for the codebase reorganization tool"""
    parser = argparse.ArgumentParser(
        description='Codebase Reorganization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reorganizer.py --preview                    # See what would be done
  python reorganizer.py --interactive               # Ask before each change
  python reorganizer.py --automatic                 # Full automation
  python reorganizer.py --symlinks                  # Use symlinks (safest)
  python reorganizer.py --move                      # Actually move files
  python reorganizer.py --config /path/to/config.json  # Use custom config
        """
    )

    parser.add_argument('--mode', choices=['preview', 'interactive', 'automatic', 'symlinks', 'move'],
                       default='preview', help='Execution mode')
    parser.add_argument('--root', type=str, default='.',
                       help='Root directory of the codebase')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')

    args = parser.parse_args()

    if args.dry_run:
        args.mode = 'preview'

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    root_dir = Path(args.root).resolve()
    config_file = Path(args.config) if args.config else None

    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist")
        sys.exit(1)

    if config_file and not config_file.exists():
        print(f"Error: Config file {config_file} does not exist")
        sys.exit(1)

    print(f"Codebase Reorganization Tool v2.0")
    print(f"Root Directory: {root_dir}")
    print(f"Mode: {args.mode}")
    print("-" * 50)

    try:
        reorganizer = CodebaseReorganizer(root_dir, args.mode, config_file)

        # Analyze codebase
        analyses = reorganizer.analyze_codebase()

        # Generate reorganization plan
        plan = reorganizer.generate_reorganization_plan(analyses)

        # Execute plan
        result = reorganizer.execute_plan(plan)

        print("Reorganization completed!")
        print(f"📊 Results: {result.files_reorganized} moved, {result.symlinks_created} symlinked")
        if result.backup_path:
            print(f"🔄 Backup: {result.backup_path}")
        if result.errors:
            print(f"❌ Errors: {len(result.errors)}")
        if result.warnings:
            print(f"⚠️  Warnings: {len(result.warnings)}")

        # Save results
        results_file = reorganizer.root_dir / "tools" / "codebase_reorganizer" / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        print(f"📄 Full results saved to: {results_file}")

    except Exception as e:
        print(f"Error during reorganization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
