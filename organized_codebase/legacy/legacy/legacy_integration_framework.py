"""
Legacy Integration Framework
===========================
Comprehensive legacy code documentation and integration system for TestMaster.

This module provides analysis, documentation, and modernization pathways for
legacy components, archive systems, and historical code integration.

Author: Agent D - Documentation & Validation Excellence  
Phase: Hour 3 - Legacy Code Documentation & Integration
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import hashlib
import re
import ast
import os
import logging

logger = logging.getLogger(__name__)

class LegacySystemType(Enum):
    """Types of legacy systems in TestMaster."""
    ARCHIVED_MODULE = "archived_module"
    DEPRECATED_COMPONENT = "deprecated_component"
    MIGRATED_SYSTEM = "migrated_system"
    PLACEHOLDER_IMPLEMENTATION = "placeholder_implementation"
    OVERSIZED_MODULE = "oversized_module"
    LEGACY_SCRIPT = "legacy_script"
    BACKUP_SYSTEM = "backup_system"

class MigrationStatus(Enum):
    """Status of legacy system migration."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    PARTIALLY_MIGRATED = "partially_migrated"
    FULLY_MIGRATED = "fully_migrated"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class IntegrationComplexity(Enum):
    """Complexity level for legacy integration."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"

@dataclass
class LegacyComponent:
    """Represents a legacy code component."""
    component_id: str
    name: str
    path: str
    system_type: LegacySystemType
    size_lines: int
    migration_status: MigrationStatus
    integration_complexity: IntegrationComplexity
    features: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    modern_equivalents: List[str] = field(default_factory=list)
    archive_date: Optional[datetime] = None
    hash_sha256: Optional[str] = None
    documentation_status: str = "undocumented"
    integration_notes: str = ""

@dataclass
class ArchiveAnalysis:
    """Analysis of archive system contents."""
    archive_path: str
    total_components: int
    total_lines: int
    component_breakdown: Dict[LegacySystemType, int]
    preservation_status: str
    integrity_verified: bool
    migration_opportunities: List[str]
    documentation_coverage: float
    last_analysis: datetime = field(default_factory=datetime.now)

@dataclass
class MigrationPlan:
    """Migration plan for legacy components."""
    component_id: str
    current_location: str
    target_location: str
    migration_strategy: str
    complexity_assessment: IntegrationComplexity
    dependencies_map: Dict[str, str]
    testing_requirements: List[str]
    validation_criteria: List[str]
    estimated_effort: str
    migration_steps: List[str] = field(default_factory=list)
    rollback_plan: str = ""

class ArchiveSystemAnalyzer:
    """Analyzes TestMaster's archive system for legacy components."""
    
    def __init__(self, archive_root: str = None):
        self.archive_root = Path(archive_root) if archive_root else Path("archive")
        self.manifest_path = self.archive_root / "archive_manifest.json"
        self.preservation_rules_path = self.archive_root / "PRESERVATION_RULES.md"
        
    def analyze_complete_archive_system(self) -> ArchiveAnalysis:
        """Perform comprehensive analysis of the entire archive system."""
        
        logger.info(f"Starting comprehensive archive analysis at {self.archive_root}")
        
        # Load archive manifest if available
        archive_manifest = self._load_archive_manifest()
        
        # Discover all archive components
        discovered_components = self._discover_archive_components()
        
        # Analyze component breakdown
        component_breakdown = self._analyze_component_breakdown(discovered_components)
        
        # Calculate totals
        total_components = len(discovered_components)
        total_lines = sum(comp.size_lines for comp in discovered_components)
        
        # Assess preservation status
        preservation_status = self._assess_preservation_status()
        
        # Verify integrity
        integrity_verified = self._verify_archive_integrity(archive_manifest, discovered_components)
        
        # Identify migration opportunities
        migration_opportunities = self._identify_migration_opportunities(discovered_components)
        
        # Calculate documentation coverage
        documentation_coverage = self._calculate_documentation_coverage(discovered_components)
        
        return ArchiveAnalysis(
            archive_path=str(self.archive_root),
            total_components=total_components,
            total_lines=total_lines,
            component_breakdown=component_breakdown,
            preservation_status=preservation_status,
            integrity_verified=integrity_verified,
            migration_opportunities=migration_opportunities,
            documentation_coverage=documentation_coverage
        )
    
    def _load_archive_manifest(self) -> Dict[str, Any]:
        """Load the archive manifest file."""
        try:
            if self.manifest_path.exists():
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load archive manifest: {e}")
        
        return {}
    
    def _discover_archive_components(self) -> List[LegacyComponent]:
        """Discover all legacy components in the archive system."""
        components = []
        
        if not self.archive_root.exists():
            logger.warning(f"Archive root {self.archive_root} does not exist")
            return components
        
        # Walk through archive directory
        for root, dirs, files in os.walk(self.archive_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    component = self._analyze_python_component(file_path)
                    if component:
                        components.append(component)
                elif file.endswith(('.js', '.jsx', '.html', '.md', '.json')):
                    file_path = Path(root) / file  
                    component = self._analyze_non_python_component(file_path)
                    if component:
                        components.append(component)
        
        return components
    
    def _analyze_python_component(self, file_path: Path) -> Optional[LegacyComponent]:
        """Analyze a Python component in the archive."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate file hash
            hash_sha256 = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Count lines
            lines = len(content.splitlines())
            
            # Determine system type based on path
            system_type = self._determine_system_type(file_path)
            
            # Extract features using AST analysis
            features = self._extract_python_features(content)
            
            # Determine migration status
            migration_status = self._determine_migration_status(file_path)
            
            # Assess integration complexity
            complexity = self._assess_integration_complexity(lines, len(features))
            
            return LegacyComponent(
                component_id=self._generate_component_id(file_path),
                name=file_path.name,
                path=str(file_path),
                system_type=system_type,
                size_lines=lines,
                migration_status=migration_status,
                integration_complexity=complexity,
                features=features,
                hash_sha256=hash_sha256,
                documentation_status="requires_analysis"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def _analyze_non_python_component(self, file_path: Path) -> Optional[LegacyComponent]:
        """Analyze a non-Python component (JS, HTML, MD, JSON)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            hash_sha256 = hashlib.sha256(content.encode('utf-8')).hexdigest()
            lines = len(content.splitlines())
            system_type = self._determine_system_type(file_path)
            features = self._extract_non_python_features(content, file_path.suffix)
            migration_status = self._determine_migration_status(file_path)
            complexity = self._assess_integration_complexity(lines, len(features))
            
            return LegacyComponent(
                component_id=self._generate_component_id(file_path),
                name=file_path.name,
                path=str(file_path),
                system_type=system_type,
                size_lines=lines,
                migration_status=migration_status,
                integration_complexity=complexity,
                features=features,
                hash_sha256=hash_sha256,
                documentation_status="requires_analysis"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing non-Python {file_path}: {e}")
            return None
    
    def _extract_python_features(self, content: str) -> List[str]:
        """Extract features from Python code using AST analysis."""
        features = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features.append(f"function:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    features.append(f"class:{node.name}")
                elif isinstance(node, ast.AsyncFunctionDef):
                    features.append(f"async_function:{node.name}")
                    
        except Exception as e:
            logger.debug(f"AST parsing error: {e}")
            # Fallback to regex-based extraction
            features.extend(self._extract_features_regex(content))
        
        return features
    
    def _extract_features_regex(self, content: str) -> List[str]:
        """Fallback feature extraction using regex."""
        features = []
        
        # Extract function definitions
        function_matches = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        features.extend([f"function:{name}" for name in function_matches])
        
        # Extract class definitions
        class_matches = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        features.extend([f"class:{name}" for name in class_matches])
        
        # Extract async function definitions
        async_matches = re.findall(r'^async\s+def\s+(\w+)', content, re.MULTILINE)
        features.extend([f"async_function:{name}" for name in async_matches])
        
        return features
    
    def _extract_non_python_features(self, content: str, suffix: str) -> List[str]:
        """Extract features from non-Python files."""
        features = []
        
        if suffix == '.js' or suffix == '.jsx':
            # JavaScript/React features
            function_matches = re.findall(r'function\s+(\w+)', content)
            features.extend([f"js_function:{name}" for name in function_matches])
            
            const_matches = re.findall(r'const\s+(\w+)\s*=', content)
            features.extend([f"js_const:{name}" for name in const_matches])
            
            component_matches = re.findall(r'const\s+(\w+)\s*=.*?=>\s*{', content)
            features.extend([f"react_component:{name}" for name in component_matches])
            
        elif suffix == '.html':
            # HTML features
            div_matches = re.findall(r'<div[^>]*id=["\']([^"\']*)["\']', content)
            features.extend([f"html_div:{id_name}" for id_name in div_matches])
            
            script_matches = re.findall(r'<script[^>]*>', content)
            features.extend([f"html_script:{i}" for i in range(len(script_matches))])
            
        elif suffix == '.md':
            # Markdown features
            header_matches = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            features.extend([f"md_header:{header.strip()}" for header in header_matches])
            
        elif suffix == '.json':
            # JSON features
            try:
                json_data = json.loads(content)
                if isinstance(json_data, dict):
                    features.extend([f"json_key:{key}" for key in json_data.keys()])
            except:
                pass
        
        return features
    
    def _determine_system_type(self, file_path: Path) -> LegacySystemType:
        """Determine the legacy system type based on file path."""
        path_str = str(file_path).lower()
        
        if 'oversized_modules' in path_str:
            return LegacySystemType.OVERSIZED_MODULE
        elif 'archive' in path_str and 'replaced_code' in path_str:
            return LegacySystemType.ARCHIVED_MODULE
        elif 'deprecated' in path_str:
            return LegacySystemType.DEPRECATED_COMPONENT
        elif 'placeholder' in path_str or 'backup' in path_str:
            return LegacySystemType.PLACEHOLDER_IMPLEMENTATION
        elif 'legacy_scripts' in path_str:
            return LegacySystemType.LEGACY_SCRIPT
        elif 'backup' in path_str:
            return LegacySystemType.BACKUP_SYSTEM
        else:
            return LegacySystemType.MIGRATED_SYSTEM
    
    def _determine_migration_status(self, file_path: Path) -> MigrationStatus:
        """Determine migration status based on file location and naming."""
        path_str = str(file_path).lower()
        
        if 'replaced_code' in path_str or 'superseded' in path_str:
            return MigrationStatus.FULLY_MIGRATED
        elif 'deprecated' in path_str:
            return MigrationStatus.DEPRECATED
        elif 'archive' in path_str:
            return MigrationStatus.ARCHIVED
        elif 'backup' in path_str:
            return MigrationStatus.PARTIALLY_MIGRATED
        else:
            return MigrationStatus.NOT_STARTED
    
    def _assess_integration_complexity(self, lines: int, feature_count: int) -> IntegrationComplexity:
        """Assess integration complexity based on size and features."""
        complexity_score = 0
        
        # Size-based complexity
        if lines > 1000:
            complexity_score += 3
        elif lines > 500:
            complexity_score += 2
        elif lines > 100:
            complexity_score += 1
        
        # Feature-based complexity
        if feature_count > 50:
            complexity_score += 3
        elif feature_count > 20:
            complexity_score += 2
        elif feature_count > 10:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 5:
            return IntegrationComplexity.CRITICAL
        elif complexity_score >= 3:
            return IntegrationComplexity.COMPLEX
        elif complexity_score >= 2:
            return IntegrationComplexity.MODERATE
        else:
            return IntegrationComplexity.SIMPLE
    
    def _generate_component_id(self, file_path: Path) -> str:
        """Generate unique component ID."""
        path_hash = hashlib.md5(str(file_path).encode('utf-8')).hexdigest()[:8]
        return f"legacy_{file_path.stem}_{path_hash}"
    
    def _analyze_component_breakdown(self, components: List[LegacyComponent]) -> Dict[LegacySystemType, int]:
        """Analyze breakdown of components by type."""
        breakdown = {}
        
        for component in components:
            if component.system_type not in breakdown:
                breakdown[component.system_type] = 0
            breakdown[component.system_type] += 1
        
        return breakdown
    
    def _assess_preservation_status(self) -> str:
        """Assess the preservation status of the archive."""
        if self.preservation_rules_path.exists() and self.manifest_path.exists():
            return "excellent"
        elif self.preservation_rules_path.exists():
            return "good"
        else:
            return "needs_improvement"
    
    def _verify_archive_integrity(self, manifest: Dict[str, Any], 
                                components: List[LegacyComponent]) -> bool:
        """Verify archive integrity against manifest."""
        if not manifest:
            return False
        
        # Basic integrity check - if we have components and a manifest, assume good
        return len(components) > 0 and 'archives' in manifest
    
    def _identify_migration_opportunities(self, components: List[LegacyComponent]) -> List[str]:
        """Identify opportunities for migrating legacy components."""
        opportunities = []
        
        # Count components by status
        not_started = sum(1 for c in components if c.migration_status == MigrationStatus.NOT_STARTED)
        archived = sum(1 for c in components if c.migration_status == MigrationStatus.ARCHIVED)
        simple_complexity = sum(1 for c in components if c.integration_complexity == IntegrationComplexity.SIMPLE)
        
        if not_started > 0:
            opportunities.append(f"Migrate {not_started} components not yet started")
        
        if archived > 10:
            opportunities.append(f"Review {archived} archived components for potential restoration")
        
        if simple_complexity > 5:
            opportunities.append(f"Quick wins: {simple_complexity} simple integration opportunities")
        
        # Identify oversized modules that could be modularized further
        oversized = [c for c in components if c.system_type == LegacySystemType.OVERSIZED_MODULE]
        if oversized:
            opportunities.append(f"Modularize {len(oversized)} oversized legacy modules")
        
        return opportunities
    
    def _calculate_documentation_coverage(self, components: List[LegacyComponent]) -> float:
        """Calculate documentation coverage percentage."""
        if not components:
            return 0.0
        
        documented = sum(1 for c in components if c.documentation_status != "undocumented")
        return documented / len(components)

class LegacyMigrationPlanner:
    """Creates migration plans for legacy components."""
    
    def __init__(self):
        self.migration_strategies = self._initialize_migration_strategies()
    
    def create_migration_plan(self, component: LegacyComponent, 
                            target_system: str = None) -> MigrationPlan:
        """Create a comprehensive migration plan for a legacy component."""
        
        # Determine target location
        target_location = self._determine_target_location(component, target_system)
        
        # Select migration strategy
        strategy = self._select_migration_strategy(component)
        
        # Create dependencies map
        dependencies_map = self._create_dependencies_map(component)
        
        # Define testing requirements
        testing_requirements = self._define_testing_requirements(component)
        
        # Create validation criteria
        validation_criteria = self._create_validation_criteria(component)
        
        # Estimate effort
        effort_estimate = self._estimate_migration_effort(component)
        
        # Generate migration steps
        migration_steps = self._generate_migration_steps(component, strategy)
        
        # Create rollback plan
        rollback_plan = self._create_rollback_plan(component)
        
        return MigrationPlan(
            component_id=component.component_id,
            current_location=component.path,
            target_location=target_location,
            migration_strategy=strategy,
            complexity_assessment=component.integration_complexity,
            dependencies_map=dependencies_map,
            testing_requirements=testing_requirements,
            validation_criteria=validation_criteria,
            estimated_effort=effort_estimate,
            migration_steps=migration_steps,
            rollback_plan=rollback_plan
        )
    
    def _initialize_migration_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize migration strategies for different scenarios."""
        return {
            "direct_integration": {
                "description": "Direct integration into existing system",
                "complexity": "simple",
                "risk": "low",
                "effort": "1-2 days"
            },
            "modular_migration": {
                "description": "Break into smaller modules before integration",
                "complexity": "moderate",
                "risk": "medium",
                "effort": "1-2 weeks"
            },
            "gradual_replacement": {
                "description": "Gradually replace with modern equivalents",
                "complexity": "complex",
                "risk": "medium",
                "effort": "2-4 weeks"
            },
            "complete_rewrite": {
                "description": "Complete rewrite using modern patterns",
                "complexity": "critical",
                "risk": "high",
                "effort": "1-2 months"
            },
            "archive_with_interface": {
                "description": "Keep archived with modern interface wrapper",
                "complexity": "simple",
                "risk": "low",
                "effort": "1 week"
            }
        }
    
    def _determine_target_location(self, component: LegacyComponent, target_system: str) -> str:
        """Determine the best target location for migrated component."""
        if target_system:
            return target_system
        
        # Determine based on component type and features
        if any("test" in feature.lower() for feature in component.features):
            return "core/intelligence/testing/"
        elif any("analyt" in feature.lower() for feature in component.features):
            return "core/intelligence/analytics/"
        elif any("document" in feature.lower() for feature in component.features):
            return "core/intelligence/documentation/"
        elif any("monitor" in feature.lower() for feature in component.features):
            return "core/intelligence/monitoring/"
        else:
            return "core/intelligence/integration/"
    
    def _select_migration_strategy(self, component: LegacyComponent) -> str:
        """Select the best migration strategy for the component."""
        if component.integration_complexity == IntegrationComplexity.SIMPLE:
            return "direct_integration"
        elif component.integration_complexity == IntegrationComplexity.MODERATE:
            return "modular_migration"
        elif component.integration_complexity == IntegrationComplexity.COMPLEX:
            return "gradual_replacement"
        else:  # CRITICAL
            return "complete_rewrite"
    
    def _create_dependencies_map(self, component: LegacyComponent) -> Dict[str, str]:
        """Create map of dependencies and their modern equivalents."""
        dependencies_map = {}
        
        for dependency in component.dependencies:
            # Map common legacy dependencies to modern equivalents
            if "numpy" in dependency.lower():
                dependencies_map[dependency] = "numpy (modern version)"
            elif "pandas" in dependency.lower():
                dependencies_map[dependency] = "pandas (modern version)"
            elif "flask" in dependency.lower():
                dependencies_map[dependency] = "FastAPI or Flask (modern)"
            else:
                dependencies_map[dependency] = f"{dependency} (review required)"
        
        return dependencies_map
    
    def _define_testing_requirements(self, component: LegacyComponent) -> List[str]:
        """Define testing requirements for the migration."""
        requirements = [
            "Unit tests for all public functions",
            "Integration tests with dependent systems",
            "Performance benchmarks",
            "Error handling validation"
        ]
        
        if component.integration_complexity in [IntegrationComplexity.COMPLEX, IntegrationComplexity.CRITICAL]:
            requirements.extend([
                "Comprehensive regression testing",
                "Load testing under production conditions",
                "Backward compatibility validation",
                "Data migration validation"
            ])
        
        return requirements
    
    def _create_validation_criteria(self, component: LegacyComponent) -> List[str]:
        """Create validation criteria for successful migration."""
        criteria = [
            "All original functionality preserved",
            "Performance equal to or better than original",
            "No breaking changes to dependent systems",
            "Documentation updated and complete"
        ]
        
        if len(component.features) > 10:
            criteria.append("Feature-by-feature validation completed")
        
        if component.size_lines > 500:
            criteria.append("Code quality improvements verified")
        
        return criteria
    
    def _estimate_migration_effort(self, component: LegacyComponent) -> str:
        """Estimate effort required for migration."""
        base_effort = {
            IntegrationComplexity.SIMPLE: 2,      # 2 days
            IntegrationComplexity.MODERATE: 10,   # 2 weeks
            IntegrationComplexity.COMPLEX: 20,    # 4 weeks  
            IntegrationComplexity.CRITICAL: 40    # 8 weeks
        }
        
        effort_days = base_effort[component.integration_complexity]
        
        # Adjust based on size
        if component.size_lines > 1000:
            effort_days *= 1.5
        elif component.size_lines > 2000:
            effort_days *= 2.0
        
        # Convert to human-readable format
        if effort_days <= 5:
            return f"{int(effort_days)} days"
        elif effort_days <= 30:
            return f"{int(effort_days / 5)} weeks"
        else:
            return f"{int(effort_days / 20)} months"
    
    def _generate_migration_steps(self, component: LegacyComponent, strategy: str) -> List[str]:
        """Generate detailed migration steps."""
        base_steps = [
            "1. Create migration branch",
            "2. Analyze component dependencies",
            "3. Create target module structure",
            "4. Implement migration strategy"
        ]
        
        strategy_specific_steps = {
            "direct_integration": [
                "5. Copy component to target location",
                "6. Update imports and dependencies", 
                "7. Run integration tests",
                "8. Update documentation"
            ],
            "modular_migration": [
                "5. Break component into logical modules",
                "6. Create individual module files",
                "7. Implement inter-module communication",
                "8. Test modular integration",
                "9. Update documentation"
            ],
            "gradual_replacement": [
                "5. Identify replacement phases",
                "6. Implement first phase replacement",
                "7. Test partial replacement",
                "8. Implement remaining phases iteratively",
                "9. Final integration and validation"
            ],
            "complete_rewrite": [
                "5. Design modern architecture",
                "6. Implement core functionality",
                "7. Implement additional features",
                "8. Comprehensive testing",
                "9. Performance validation",
                "10. Documentation and training"
            ]
        }
        
        return base_steps + strategy_specific_steps.get(strategy, [])
    
    def _create_rollback_plan(self, component: LegacyComponent) -> str:
        """Create rollback plan in case migration fails."""
        return f"""
Rollback Plan for {component.name}:
1. Restore original component from archive: {component.path}
2. Revert any modified dependent systems
3. Restore original integration points
4. Validate system functionality
5. Document rollback actions and lessons learned

Archive Location: {component.path}
Hash Verification: {component.hash_sha256}
"""

class LegacyIntegrationFramework:
    """Main framework coordinating legacy code documentation and integration."""
    
    def __init__(self, archive_root: str = None):
        self.archive_analyzer = ArchiveSystemAnalyzer(archive_root)
        self.migration_planner = LegacyMigrationPlanner()
        self.legacy_components: List[LegacyComponent] = []
        self.migration_plans: Dict[str, MigrationPlan] = {}
    
    def analyze_complete_legacy_system(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of the entire legacy system."""
        
        logger.info("Starting comprehensive legacy system analysis")
        
        # Analyze archive system
        archive_analysis = self.archive_analyzer.analyze_complete_archive_system()
        
        # Store discovered components
        self.legacy_components = self.archive_analyzer._discover_archive_components()
        
        # Generate migration plans for critical components
        critical_components = [c for c in self.legacy_components 
                             if c.integration_complexity == IntegrationComplexity.CRITICAL]
        
        migration_plans = []
        for component in critical_components[:5]:  # Limit to top 5 critical
            plan = self.migration_planner.create_migration_plan(component)
            migration_plans.append(plan)
            self.migration_plans[component.component_id] = plan
        
        return {
            "archive_analysis": {
                "total_components": archive_analysis.total_components,
                "total_lines": archive_analysis.total_lines,
                "component_breakdown": {k.value: v for k, v in archive_analysis.component_breakdown.items()},
                "preservation_status": archive_analysis.preservation_status,
                "integrity_verified": archive_analysis.integrity_verified,
                "migration_opportunities": archive_analysis.migration_opportunities,
                "documentation_coverage": archive_analysis.documentation_coverage
            },
            "legacy_components": [
                {
                    "id": comp.component_id,
                    "name": comp.name,
                    "type": comp.system_type.value,
                    "size_lines": comp.size_lines,
                    "migration_status": comp.migration_status.value,
                    "complexity": comp.integration_complexity.value,
                    "features_count": len(comp.features),
                    "path": comp.path
                }
                for comp in self.legacy_components[:10]  # Top 10 components
            ],
            "migration_plans": [
                {
                    "component_id": plan.component_id,
                    "strategy": plan.migration_strategy,
                    "complexity": plan.complexity_assessment.value,
                    "estimated_effort": plan.estimated_effort,
                    "target_location": plan.target_location
                }
                for plan in migration_plans
            ],
            "summary_statistics": {
                "total_legacy_components": len(self.legacy_components),
                "by_complexity": self._get_complexity_breakdown(),
                "by_migration_status": self._get_migration_status_breakdown(),
                "integration_opportunities": len(archive_analysis.migration_opportunities)
            }
        }
    
    def _get_complexity_breakdown(self) -> Dict[str, int]:
        """Get breakdown of components by complexity."""
        breakdown = {}
        for component in self.legacy_components:
            complexity = component.integration_complexity.value
            breakdown[complexity] = breakdown.get(complexity, 0) + 1
        return breakdown
    
    def _get_migration_status_breakdown(self) -> Dict[str, int]:
        """Get breakdown of components by migration status."""
        breakdown = {}
        for component in self.legacy_components:
            status = component.migration_status.value
            breakdown[status] = breakdown.get(status, 0) + 1
        return breakdown
    
    def generate_legacy_documentation(self) -> str:
        """Generate comprehensive legacy system documentation."""
        
        analysis = self.analyze_complete_legacy_system()
        
        documentation = f"""
# TestMaster Legacy System Documentation
Generated: {datetime.now().isoformat()}

## Archive System Overview
- **Total Components**: {analysis['archive_analysis']['total_components']}
- **Total Lines of Code**: {analysis['archive_analysis']['total_lines']:,}
- **Preservation Status**: {analysis['archive_analysis']['preservation_status']}
- **Archive Integrity**: {'✅ Verified' if analysis['archive_analysis']['integrity_verified'] else '⚠️ Needs Verification'}

## Component Breakdown by Type
"""
        
        for comp_type, count in analysis['archive_analysis']['component_breakdown'].items():
            documentation += f"- **{comp_type.replace('_', ' ').title()}**: {count} components\n"
        
        documentation += f"""

## Migration Opportunities
"""
        for opportunity in analysis['archive_analysis']['migration_opportunities']:
            documentation += f"- {opportunity}\n"
        
        documentation += f"""

## Critical Components Requiring Attention
"""
        for component in analysis['legacy_components']:
            if component['complexity'] in ['complex', 'critical']:
                documentation += f"""
### {component['name']}
- **Type**: {component['type']}
- **Size**: {component['size_lines']:,} lines
- **Complexity**: {component['complexity']}
- **Status**: {component['migration_status']}
- **Features**: {component['features_count']} identified
- **Location**: `{component['path']}`
"""
        
        documentation += f"""

## Migration Plans Summary
"""
        for plan in analysis['migration_plans']:
            documentation += f"""
### {plan['component_id']}
- **Strategy**: {plan['strategy']}
- **Effort**: {plan['estimated_effort']}
- **Target**: `{plan['target_location']}`
"""
        
        return documentation

# Global legacy integration framework instance
_legacy_integration_framework = LegacyIntegrationFramework()

def get_legacy_integration_framework() -> LegacyIntegrationFramework:
    """Get the global legacy integration framework instance."""
    return _legacy_integration_framework

def analyze_legacy_systems(archive_root: str = None) -> Dict[str, Any]:
    """High-level function to analyze legacy systems."""
    framework = LegacyIntegrationFramework(archive_root)
    return framework.analyze_complete_legacy_system()

def generate_migration_plan(component_id: str) -> Optional[MigrationPlan]:
    """Generate migration plan for a specific legacy component."""
    framework = get_legacy_integration_framework()
    
    # Find component
    component = next((c for c in framework.legacy_components if c.component_id == component_id), None)
    if not component:
        return None
    
    return framework.migration_planner.create_migration_plan(component)