"""
Automatic File Tagging and Classification System

Inspired by Agent-Squad's configuration-driven classification
and Agency-Swarm's intelligent categorization patterns.

Features:
- Automatic module type classification (core, utility, test, config)
- Status tags (stable, breaking, needs-attention, idle)
- Priority levels for Claude Code attention
- Dynamic tag updates based on analysis results
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from core.layer_manager import requires_layer


class ModuleType(Enum):
    """Types of modules in the codebase."""
    CORE = "core"
    UTILITY = "utility"
    API = "api"
    UI = "ui"
    TEST = "test"
    CONFIG = "config"
    SCRIPT = "script"
    DATA = "data"
    UNKNOWN = "unknown"


class ModuleStatus(Enum):
    """Status of modules."""
    STABLE = "stable"
    BREAKING = "breaking"
    NEEDS_ATTENTION = "needs_attention"
    IDLE = "idle"
    UNDER_DEVELOPMENT = "under_development"
    DEPRECATED = "deprecated"


class Priority(Enum):
    """Priority levels for Claude Code attention."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class FileClassification:
    """Classification result for a file."""
    file_path: str
    module_type: ModuleType
    status: ModuleStatus
    priority: Priority
    
    # Analysis results
    complexity_score: float
    importance_score: float
    stability_score: float
    
    # Metadata
    line_count: int
    class_count: int
    function_count: int
    import_count: int
    test_coverage: Optional[float] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    
    # Timestamps
    last_modified: datetime = field(default_factory=datetime.now)
    last_analyzed: datetime = field(default_factory=datetime.now)
    
    # Generated tags
    tags: Set[str] = field(default_factory=set)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TaggingRule:
    """Rule for automatic tagging."""
    rule_id: str
    pattern: str  # Pattern to match (regex or keyword)
    target_tags: List[str]
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0  # Higher priority rules applied first


class FileTagger:
    """
    Automatic file tagging and classification system.
    
    Uses Agent-Squad configuration-driven patterns for intelligent
    file classification and Agency-Swarm categorization logic.
    """
    
    @requires_layer("layer3_orchestration", "auto_tagging")
    def __init__(self, watch_paths: Union[str, List[str]]):
        """
        Initialize file tagger.
        
        Args:
            watch_paths: Directories to monitor and tag
        """
        self.watch_paths = [Path(p) for p in (watch_paths if isinstance(watch_paths, list) else [watch_paths])]
        
        # Classification cache
        self._classifications: Dict[str, FileClassification] = {}
        self._tagging_rules: Dict[str, TaggingRule] = {}
        
        # Analysis patterns
        self._setup_default_patterns()
        
        # Statistics
        self._stats = {
            'files_analyzed': 0,
            'classifications_updated': 0,
            'tags_applied': 0,
            'last_scan': None
        }
        
        print(f"🏷️ File tagger initialized")
        print(f"   📁 Watching: {', '.join(str(p) for p in self.watch_paths)}")
    
    def _setup_default_patterns(self):
        """Setup default tagging patterns and rules."""
        
        # Core module patterns
        self.add_tagging_rule(
            "core_modules",
            r"(core|engine|main|base|foundation)",
            ["core", "important", "stable"],
            conditions={"min_lines": 50}
        )
        
        # API module patterns
        self.add_tagging_rule(
            "api_modules", 
            r"(api|endpoint|route|handler|controller)",
            ["api", "public_interface", "needs_tests"],
            conditions={"has_classes": True}
        )
        
        # Utility module patterns
        self.add_tagging_rule(
            "utility_modules",
            r"(util|helper|common|shared|tool)",
            ["utility", "reusable", "stable"],
            conditions={"function_ratio": 0.7}  # Mostly functions
        )
        
        # Test module patterns
        self.add_tagging_rule(
            "test_modules",
            r"(test_|_test\.py|tests/)",
            ["test", "automated", "coverage"],
            conditions={"is_test_file": True}
        )
        
        # Configuration patterns
        self.add_tagging_rule(
            "config_modules",
            r"(config|setting|constant|env)",
            ["config", "environment", "critical"],
            conditions={"has_constants": True}
        )
        
        # Complex module patterns
        self.add_tagging_rule(
            "complex_modules",
            r".*",  # Any file
            ["complex", "needs_review"],
            conditions={"complexity_score": "> 80"}
        )
        
        # Idle module patterns
        self.add_tagging_rule(
            "idle_modules",
            r".*",
            ["idle", "needs_attention"],
            conditions={"last_modified": "> 168"}  # >7 days
        )
    
    def add_tagging_rule(self, rule_id: str, pattern: str, target_tags: List[str],
                        conditions: Dict[str, Any] = None, priority: int = 0) -> bool:
        """
        Add a tagging rule.
        
        Args:
            rule_id: Unique rule identifier
            pattern: Pattern to match (regex or keyword)
            target_tags: Tags to apply when rule matches
            conditions: Additional conditions for rule application
            priority: Rule priority (higher = applied first)
            
        Returns:
            True if rule was added
        """
        rule = TaggingRule(
            rule_id=rule_id,
            pattern=pattern,
            target_tags=target_tags,
            conditions=conditions or {},
            priority=priority
        )
        
        self._tagging_rules[rule_id] = rule
        print(f"📋 Added tagging rule: {rule_id}")
        return True
    
    def classify_file(self, file_path: Union[str, Path]) -> Optional[FileClassification]:
        """
        Classify a single file.
        
        Args:
            file_path: Path to file to classify
            
        Returns:
            FileClassification if successful, None otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists() or not file_path.suffix == '.py':
            return None
        
        try:
            # Analyze file structure
            analysis = self._analyze_file_structure(file_path)
            
            # Determine module type
            module_type = self._classify_module_type(file_path, analysis)
            
            # Determine status
            status = self._determine_module_status(file_path, analysis)
            
            # Calculate priority
            priority = self._calculate_priority(file_path, analysis, module_type, status)
            
            # Calculate scores
            complexity_score = self._calculate_complexity_score(analysis)
            importance_score = self._calculate_importance_score(file_path, analysis, module_type)
            stability_score = self._calculate_stability_score(file_path, analysis)
            
            # Get dependencies
            dependencies = self._extract_dependencies(analysis)
            
            # Create classification
            classification = FileClassification(
                file_path=str(file_path),
                module_type=module_type,
                status=status,
                priority=priority,
                complexity_score=complexity_score,
                importance_score=importance_score,
                stability_score=stability_score,
                line_count=analysis['line_count'],
                class_count=analysis['class_count'],
                function_count=analysis['function_count'],
                import_count=analysis['import_count'],
                dependencies=dependencies,
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                last_analyzed=datetime.now()
            )
            
            # Apply tagging rules
            self._apply_tagging_rules(classification, analysis)
            
            # Generate recommendations
            self._generate_recommendations(classification, analysis)
            
            # Cache classification
            self._classifications[str(file_path)] = classification
            self._stats['files_analyzed'] += 1
            
            return classification
            
        except Exception as e:
            print(f"⚠️ Error classifying {file_path}: {e}")
            return None
    
    def classify_all_files(self, force_reclassify: bool = False) -> Dict[str, FileClassification]:
        """
        Classify all files in watch paths.
        
        Args:
            force_reclassify: Force re-classification of cached files
            
        Returns:
            Dictionary mapping file paths to classifications
        """
        print("🔍 Classifying all files...")
        
        classified_count = 0
        updated_count = 0
        
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
            
            for py_file in watch_path.rglob("*.py"):
                if self._should_classify_file(py_file):
                    classified_count += 1
                    
                    # Check if needs re-classification
                    if force_reclassify or self._needs_reclassification(py_file):
                        classification = self.classify_file(py_file)
                        if classification:
                            updated_count += 1
        
        self._stats['last_scan'] = datetime.now()
        print(f"📊 Classification complete: {classified_count} files, {updated_count} updated")
        
        return dict(self._classifications)
    
    def update_classification_from_test_results(self, test_results: Dict[str, Any]):
        """
        Update classifications based on test results.
        
        Args:
            test_results: Test execution results
        """
        for test_path, result in test_results.items():
            # Find source file for test
            source_file = self._find_source_file_for_test(test_path)
            
            if source_file and source_file in self._classifications:
                classification = self._classifications[source_file]
                
                # Update status based on test results
                if result.get('failed', False):
                    classification.status = ModuleStatus.BREAKING
                    classification.priority = Priority.HIGH
                    classification.tags.add("failing_tests")
                    classification.recommendations.append("Fix failing tests")
                elif result.get('passed', True):
                    if ModuleStatus.BREAKING in [classification.status]:
                        classification.status = ModuleStatus.STABLE
                    classification.tags.discard("failing_tests")
                
                # Update coverage
                if 'coverage' in result:
                    classification.test_coverage = result['coverage']
                    
                    if result['coverage'] < 70:
                        classification.tags.add("low_coverage")
                        classification.recommendations.append("Improve test coverage")
                    else:
                        classification.tags.discard("low_coverage")
                
                classification.last_analyzed = datetime.now()
                self._stats['classifications_updated'] += 1
    
    def update_classification_from_idle_detection(self, idle_modules: List[Dict[str, Any]]):
        """
        Update classifications based on idle module detection.
        
        Args:
            idle_modules: List of idle module information
        """
        for idle_info in idle_modules:
            module_path = idle_info.get('module_path') or idle_info.get('path')
            
            if module_path and module_path in self._classifications:
                classification = self._classifications[module_path]
                
                # Update status to idle
                classification.status = ModuleStatus.IDLE
                classification.tags.add("idle")
                classification.tags.add("needs_attention")
                
                # Adjust priority based on importance
                if classification.importance_score > 80:
                    classification.priority = Priority.HIGH
                    classification.recommendations.append("High-importance module is idle - review for updates")
                elif classification.importance_score > 50:
                    classification.priority = Priority.NORMAL
                    classification.recommendations.append("Module is idle - consider updates or testing")
                else:
                    classification.priority = Priority.LOW
                
                classification.last_analyzed = datetime.now()
                self._stats['classifications_updated'] += 1
    
    def get_files_by_type(self, module_type: ModuleType) -> List[FileClassification]:
        """Get files of a specific type."""
        return [
            classification for classification in self._classifications.values()
            if classification.module_type == module_type
        ]
    
    def get_files_by_status(self, status: ModuleStatus) -> List[FileClassification]:
        """Get files with a specific status."""
        return [
            classification for classification in self._classifications.values()
            if classification.status == status
        ]
    
    def get_files_by_priority(self, priority: Priority) -> List[FileClassification]:
        """Get files with a specific priority."""
        return [
            classification for classification in self._classifications.values()
            if classification.priority == priority
        ]
    
    def get_files_with_tag(self, tag: str) -> List[FileClassification]:
        """Get files that have a specific tag."""
        return [
            classification for classification in self._classifications.values()
            if tag in classification.tags
        ]
    
    def get_high_priority_files(self) -> List[FileClassification]:
        """Get files that need immediate attention."""
        high_priority = []
        
        for classification in self._classifications.values():
            if (classification.priority in [Priority.HIGH, Priority.CRITICAL, Priority.EMERGENCY] or
                classification.status in [ModuleStatus.BREAKING, ModuleStatus.NEEDS_ATTENTION] or
                "failing_tests" in classification.tags or
                "low_coverage" in classification.tags):
                high_priority.append(classification)
        
        # Sort by priority and importance
        high_priority.sort(key=lambda c: (-c.priority.name, -c.importance_score))
        return high_priority
    
    def generate_claude_directives(self) -> Dict[str, Any]:
        """
        Generate CLAUDE.md directives based on file classifications.
        
        Returns:
            Dictionary of directives for Claude Code
        """
        high_priority_files = self.get_high_priority_files()
        breaking_files = self.get_files_by_status(ModuleStatus.BREAKING)
        idle_files = self.get_files_by_status(ModuleStatus.IDLE)
        
        directives = {
            "monitor_priority": [],
            "immediate_actions": [],
            "test_preferences": [],
            "coverage_targets": []
        }
        
        # High priority monitoring
        for classification in high_priority_files[:10]:  # Top 10
            directives["monitor_priority"].append({
                "path": classification.file_path,
                "level": classification.priority.value.upper(),
                "reason": f"{classification.module_type.value} module with {classification.status.value} status",
                "recommendations": classification.recommendations
            })
        
        # Immediate actions for breaking files
        for classification in breaking_files:
            directives["immediate_actions"].append(
                f"Fix breaking {classification.module_type.value} module: {Path(classification.file_path).name}"
            )
        
        # Test preferences based on module types
        api_files = self.get_files_by_type(ModuleType.API)
        if api_files:
            directives["test_preferences"].append({
                "module_pattern": "*/api/*",
                "test_style": "integration_first",
                "coverage_target": 90,
                "reason": "API modules need comprehensive testing"
            })
        
        core_files = self.get_files_by_type(ModuleType.CORE)
        if core_files:
            directives["test_preferences"].append({
                "module_pattern": "*/core/*",
                "test_style": "unit_comprehensive",
                "coverage_target": 95,
                "reason": "Core modules are critical and need thorough testing"
            })
        
        # Coverage targets for low coverage files
        low_coverage_files = self.get_files_with_tag("low_coverage")
        for classification in low_coverage_files:
            directives["coverage_targets"].append({
                "module": classification.file_path,
                "current_coverage": classification.test_coverage or 0,
                "target_coverage": 80,
                "priority": classification.priority.value
            })
        
        return directives
    
    def _analyze_file_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file structure and extract metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Count elements
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            # Count lines (excluding empty and comments)
            lines = [line.strip() for line in content.split('\\n')]
            code_lines = [line for line in lines if line and not line.startswith('#')]
            
            # Check for constants
            has_constants = bool([
                node for node in ast.walk(tree) 
                if isinstance(node, ast.Assign) and 
                any(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets)
            ])
            
            # Calculate ratios
            total_elements = len(classes) + len(functions)
            function_ratio = len(functions) / max(total_elements, 1)
            
            return {
                'content': content,
                'tree': tree,
                'line_count': len(code_lines),
                'class_count': len(classes),
                'function_count': len(functions),
                'import_count': len(imports),
                'has_constants': has_constants,
                'function_ratio': function_ratio,
                'classes': classes,
                'functions': functions,
                'imports': imports
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
            return {
                'line_count': 0,
                'class_count': 0,
                'function_count': 0,
                'import_count': 0,
                'has_constants': False,
                'function_ratio': 0.0,
                'classes': [],
                'functions': [],
                'imports': []
            }
    
    def _classify_module_type(self, file_path: Path, analysis: Dict[str, Any]) -> ModuleType:
        """Classify module type based on file analysis."""
        path_str = str(file_path).lower()
        name = file_path.name.lower()
        
        # Test files
        if (name.startswith('test_') or name.endswith('_test.py') or 
            'test' in str(file_path.parent).lower()):
            return ModuleType.TEST
        
        # Configuration files
        if any(keyword in path_str for keyword in ['config', 'setting', 'constant', 'env']):
            return ModuleType.CONFIG
        
        # API files
        if any(keyword in path_str for keyword in ['api', 'endpoint', 'route', 'handler', 'controller']):
            return ModuleType.API
        
        # UI files
        if any(keyword in path_str for keyword in ['ui', 'view', 'template', 'frontend', 'gui']):
            return ModuleType.UI
        
        # Core files
        if any(keyword in path_str for keyword in ['core', 'engine', 'main', 'base', 'foundation']):
            return ModuleType.CORE
        
        # Utility files (high function ratio)
        if (any(keyword in path_str for keyword in ['util', 'helper', 'common', 'shared', 'tool']) or
            analysis.get('function_ratio', 0) > 0.8):
            return ModuleType.UTILITY
        
        # Script files
        if (name.endswith('_script.py') or name.startswith('run_') or 
            analysis.get('class_count', 0) == 0 and analysis.get('function_count', 0) < 3):
            return ModuleType.SCRIPT
        
        # Data files
        if any(keyword in path_str for keyword in ['data', 'model', 'schema', 'entity']):
            return ModuleType.DATA
        
        return ModuleType.UNKNOWN
    
    def _determine_module_status(self, file_path: Path, analysis: Dict[str, Any]) -> ModuleStatus:
        """Determine module status based on analysis."""
        # Check if file was modified recently
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        days_since_modified = (datetime.now() - last_modified).days
        
        # Idle if not modified for a week
        if days_since_modified > 7:
            return ModuleStatus.IDLE
        
        # Under development if modified very recently
        if days_since_modified < 1:
            return ModuleStatus.UNDER_DEVELOPMENT
        
        # Default to stable
        return ModuleStatus.STABLE
    
    def _calculate_priority(self, file_path: Path, analysis: Dict[str, Any], 
                          module_type: ModuleType, status: ModuleStatus) -> Priority:
        """Calculate priority for Claude Code attention."""
        priority_score = 0
        
        # Module type priority
        type_priorities = {
            ModuleType.CORE: 40,
            ModuleType.API: 30,
            ModuleType.UI: 20,
            ModuleType.UTILITY: 15,
            ModuleType.DATA: 15,
            ModuleType.TEST: 10,
            ModuleType.CONFIG: 25,
            ModuleType.SCRIPT: 5,
            ModuleType.UNKNOWN: 5
        }
        priority_score += type_priorities.get(module_type, 0)
        
        # Status priority
        status_priorities = {
            ModuleStatus.BREAKING: 50,
            ModuleStatus.NEEDS_ATTENTION: 30,
            ModuleStatus.UNDER_DEVELOPMENT: 20,
            ModuleStatus.IDLE: 15,
            ModuleStatus.STABLE: 10,
            ModuleStatus.DEPRECATED: 5
        }
        priority_score += status_priorities.get(status, 0)
        
        # Size/complexity priority
        if analysis.get('line_count', 0) > 500:
            priority_score += 10
        if analysis.get('class_count', 0) > 5:
            priority_score += 10
        
        # Convert to priority enum
        if priority_score >= 80:
            return Priority.EMERGENCY
        elif priority_score >= 60:
            return Priority.CRITICAL
        elif priority_score >= 40:
            return Priority.HIGH
        elif priority_score >= 20:
            return Priority.NORMAL
        else:
            return Priority.LOW
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate complexity score (0-100)."""
        score = 0
        
        # Line count factor
        line_count = analysis.get('line_count', 0)
        score += min(line_count / 10, 30)  # Max 30 points for lines
        
        # Class complexity
        class_count = analysis.get('class_count', 0)
        score += min(class_count * 5, 25)  # Max 25 points for classes
        
        # Function complexity
        function_count = analysis.get('function_count', 0)
        score += min(function_count * 2, 20)  # Max 20 points for functions
        
        # Import complexity
        import_count = analysis.get('import_count', 0)
        score += min(import_count * 1, 15)  # Max 15 points for imports
        
        # Nesting complexity (simplified)
        if 'tree' in analysis:
            nested_depth = self._calculate_nesting_depth(analysis['tree'])
            score += min(nested_depth * 2, 10)  # Max 10 points for nesting
        
        return min(score, 100)
    
    def _calculate_importance_score(self, file_path: Path, analysis: Dict[str, Any], 
                                  module_type: ModuleType) -> float:
        """Calculate importance score (0-100)."""
        score = 0
        
        # Module type importance
        type_scores = {
            ModuleType.CORE: 40,
            ModuleType.API: 35,
            ModuleType.CONFIG: 30,
            ModuleType.DATA: 25,
            ModuleType.UI: 20,
            ModuleType.UTILITY: 15,
            ModuleType.TEST: 10,
            ModuleType.SCRIPT: 5,
            ModuleType.UNKNOWN: 5
        }
        score += type_scores.get(module_type, 0)
        
        # Size indicates importance
        line_count = analysis.get('line_count', 0)
        score += min(line_count / 20, 25)  # Max 25 points
        
        # Classes indicate structure and importance
        class_count = analysis.get('class_count', 0)
        score += min(class_count * 5, 20)  # Max 20 points
        
        # Location in hierarchy
        path_depth = len(file_path.parts)
        if path_depth <= 3:  # Shallow = important
            score += 15
        elif path_depth <= 5:
            score += 10
        else:
            score += 5
        
        return min(score, 100)
    
    def _calculate_stability_score(self, file_path: Path, analysis: Dict[str, Any]) -> float:
        """Calculate stability score (0-100)."""
        score = 50  # Start neutral
        
        # Age indicates stability
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        days_old = (datetime.now() - last_modified).days
        
        if days_old > 30:
            score += 25  # Old = stable
        elif days_old > 7:
            score += 15
        elif days_old < 1:
            score -= 20  # Very new = unstable
        
        # Size can indicate stability
        line_count = analysis.get('line_count', 0)
        if line_count > 200:
            score += 15  # Large = established
        elif line_count < 50:
            score -= 10  # Small = potentially unstable
        
        # Well-structured code is more stable
        if analysis.get('class_count', 0) > 0 and analysis.get('function_count', 0) > 0:
            score += 10
        
        return max(0, min(score, 100))
    
    def _extract_dependencies(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract module dependencies from imports."""
        dependencies = []
        
        for import_node in analysis.get('imports', []):
            if isinstance(import_node, ast.Import):
                for alias in import_node.names:
                    dependencies.append(alias.name)
            elif isinstance(import_node, ast.ImportFrom):
                if import_node.module:
                    dependencies.append(import_node.module)
        
        return dependencies
    
    def _apply_tagging_rules(self, classification: FileClassification, analysis: Dict[str, Any]):
        """Apply tagging rules to a classification."""
        file_path = Path(classification.file_path)
        
        # Sort rules by priority
        sorted_rules = sorted(self._tagging_rules.values(), key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # Check pattern match
            pattern_match = False
            
            # Check file path pattern
            if re.search(rule.pattern, str(file_path), re.IGNORECASE):
                pattern_match = True
            
            # Check file name pattern
            if re.search(rule.pattern, file_path.name, re.IGNORECASE):
                pattern_match = True
            
            if not pattern_match:
                continue
            
            # Check conditions
            conditions_met = True
            
            for condition, value in rule.conditions.items():
                if condition == "min_lines":
                    if analysis.get('line_count', 0) < value:
                        conditions_met = False
                elif condition == "has_classes":
                    if value and analysis.get('class_count', 0) == 0:
                        conditions_met = False
                elif condition == "function_ratio":
                    if analysis.get('function_ratio', 0) < value:
                        conditions_met = False
                elif condition == "is_test_file":
                    is_test = classification.module_type == ModuleType.TEST
                    if value != is_test:
                        conditions_met = False
                elif condition == "has_constants":
                    if value and not analysis.get('has_constants', False):
                        conditions_met = False
                elif condition == "complexity_score":
                    if isinstance(value, str) and value.startswith("> "):
                        threshold = float(value[2:])
                        if classification.complexity_score <= threshold:
                            conditions_met = False
                elif condition == "last_modified":
                    if isinstance(value, str) and value.startswith("> "):
                        hours_threshold = float(value[2:])
                        hours_old = (datetime.now() - classification.last_modified).total_seconds() / 3600
                        if hours_old <= hours_threshold:
                            conditions_met = False
            
            if conditions_met:
                # Apply tags
                for tag in rule.target_tags:
                    classification.tags.add(tag)
                    self._stats['tags_applied'] += 1
    
    def _generate_recommendations(self, classification: FileClassification, analysis: Dict[str, Any]):
        """Generate recommendations for a file."""
        recommendations = []
        
        # Complexity recommendations
        if classification.complexity_score > 80:
            recommendations.append("Consider refactoring to reduce complexity")
        
        # Coverage recommendations
        if classification.test_coverage is not None and classification.test_coverage < 70:
            recommendations.append("Improve test coverage")
        
        # Status-based recommendations
        if classification.status == ModuleStatus.IDLE:
            if classification.importance_score > 70:
                recommendations.append("Important module is idle - review for updates")
            else:
                recommendations.append("Consider if this module is still needed")
        
        elif classification.status == ModuleStatus.BREAKING:
            recommendations.append("Fix breaking issues immediately")
        
        # Type-based recommendations
        if classification.module_type == ModuleType.API and analysis.get('class_count', 0) == 0:
            recommendations.append("Consider organizing API code into classes")
        
        if classification.module_type == ModuleType.CORE and classification.stability_score < 70:
            recommendations.append("Core module needs stabilization")
        
        classification.recommendations.extend(recommendations)
    
    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth in AST."""
        max_depth = 0
        
        def get_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    get_depth(child, current_depth + 1)
                else:
                    get_depth(child, current_depth)
        
        get_depth(tree)
        return max_depth
    
    def _should_classify_file(self, file_path: Path) -> bool:
        """Check if file should be classified."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return False
        
        # Skip common ignore patterns
        ignore_patterns = {
            '__pycache__', '.git', '.vscode', '.idea', 'venv', '.env',
            'node_modules', '.pytest_cache', '.coverage', '.tox'
        }
        
        if any(pattern in str(file_path) for pattern in ignore_patterns):
            return False
        
        return True
    
    def _needs_reclassification(self, file_path: Path) -> bool:
        """Check if file needs re-classification."""
        file_str = str(file_path)
        
        # Check if never classified
        if file_str not in self._classifications:
            return True
        
        # Check if file modified since last classification
        classification = self._classifications[file_str]
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        return file_mtime > classification.last_analyzed
    
    def _find_source_file_for_test(self, test_path: str) -> Optional[str]:
        """Find the source file that corresponds to a test file."""
        test_path = Path(test_path)
        
        if not test_path.name.startswith('test_'):
            return None
        
        # Extract module name from test file
        module_name = test_path.name[5:]  # Remove 'test_' prefix
        
        # Search for corresponding source file
        for watch_path in self.watch_paths:
            for py_file in watch_path.rglob(module_name):
                if py_file.is_file() and str(py_file) in self._classifications:
                    return str(py_file)
        
        return None
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get file classification statistics."""
        total_files = len(self._classifications)
        
        # Count by type
        type_counts = {}
        for module_type in ModuleType:
            type_counts[module_type.value] = len(self.get_files_by_type(module_type))
        
        # Count by status
        status_counts = {}
        for status in ModuleStatus:
            status_counts[status.value] = len(self.get_files_by_status(status))
        
        # Count by priority
        priority_counts = {}
        for priority in Priority:
            priority_counts[priority.value] = len(self.get_files_by_priority(priority))
        
        # Calculate averages
        if total_files > 0:
            avg_complexity = sum(c.complexity_score for c in self._classifications.values()) / total_files
            avg_importance = sum(c.importance_score for c in self._classifications.values()) / total_files
            avg_stability = sum(c.stability_score for c in self._classifications.values()) / total_files
        else:
            avg_complexity = avg_importance = avg_stability = 0.0
        
        return {
            "total_files": total_files,
            "type_distribution": type_counts,
            "status_distribution": status_counts,
            "priority_distribution": priority_counts,
            "average_scores": {
                "complexity": avg_complexity,
                "importance": avg_importance,
                "stability": avg_stability
            },
            "high_priority_count": len(self.get_high_priority_files()),
            "tagging_rules": len(self._tagging_rules),
            "statistics": dict(self._stats)
        }
    
    def export_classification_report(self, output_path: str = "classification_report.json"):
        """Export comprehensive classification report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_classification_statistics(),
            "classifications": [],
            "claude_directives": self.generate_claude_directives()
        }
        
        # Add detailed classifications
        for classification in self._classifications.values():
            class_data = {
                "file_path": classification.file_path,
                "module_type": classification.module_type.value,
                "status": classification.status.value,
                "priority": classification.priority.value,
                "scores": {
                    "complexity": classification.complexity_score,
                    "importance": classification.importance_score,
                    "stability": classification.stability_score
                },
                "metrics": {
                    "line_count": classification.line_count,
                    "class_count": classification.class_count,
                    "function_count": classification.function_count,
                    "import_count": classification.import_count
                },
                "tags": list(classification.tags),
                "recommendations": classification.recommendations,
                "last_modified": classification.last_modified.isoformat(),
                "last_analyzed": classification.last_analyzed.isoformat()
            }
            report["classifications"].append(class_data)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Classification report exported to {output_path}")
        except Exception as e:
            print(f"⚠️ Error exporting classification report: {e}")


# Convenience functions for file tagging
def classify_directory(directory: str) -> Dict[str, FileClassification]:
    """Quick classification of a directory."""
    tagger = FileTagger(directory)
    return tagger.classify_all_files()


def get_high_priority_files_in_directory(directory: str) -> List[FileClassification]:
    """Get high priority files in a directory."""
    tagger = FileTagger(directory)
    tagger.classify_all_files()
    return tagger.get_high_priority_files()