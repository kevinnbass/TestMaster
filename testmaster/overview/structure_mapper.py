"""
Functional Structure Mapping

Comprehensive codebase structure analysis and relationship mapping
for understanding module dependencies, API surfaces, and business logic.

Features:
- Module relationship graphs with dependency analysis
- API surface tracking and public interface identification
- Business logic identification and categorization
- Architectural pattern detection and documentation
"""

import ast
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import networkx as nx

from ..core.layer_manager import requires_layer
from ..core.feature_flags import FeatureFlags
from .performance_dashboard import get_performance_dashboard, record_dashboard_metric, MetricType


class RelationshipType(Enum):
    """Types of module relationships."""
    IMPORTS = "imports"
    INHERITS = "inherits"
    CALLS = "calls"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    AGGREGATES = "aggregates"


class ModuleCategory(Enum):
    """Categories of modules based on functionality."""
    CORE_BUSINESS = "core_business"
    API_LAYER = "api_layer"
    DATA_ACCESS = "data_access"
    UTILITY = "utility"
    INFRASTRUCTURE = "infrastructure"
    USER_INTERFACE = "user_interface"
    CONFIGURATION = "configuration"
    TESTING = "testing"
    EXTERNAL_INTEGRATION = "external_integration"


class AccessLevel(Enum):
    """Access levels for classes and functions."""
    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"
    INTERNAL = "internal"


@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    module_path: str
    line_number: int
    access_level: AccessLevel
    
    # Function characteristics
    is_async: bool = False
    is_method: bool = False
    is_staticmethod: bool = False
    is_classmethod: bool = False
    is_property: bool = False
    
    # Function metadata
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    
    # Complexity metrics
    cyclomatic_complexity: int = 1
    line_count: int = 0
    
    # Dependencies
    calls_functions: List[str] = field(default_factory=list)
    calls_external: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    module_path: str
    line_number: int
    access_level: AccessLevel
    
    # Class characteristics
    is_abstract: bool = False
    is_dataclass: bool = False
    is_enum: bool = False
    is_exception: bool = False
    
    # Class metadata
    docstring: Optional[str] = None
    base_classes: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    
    # Class contents
    methods: List[FunctionInfo] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    
    # Metrics
    method_count: int = 0
    line_count: int = 0


@dataclass
class ModuleInfo:
    """Information about a module."""
    module_path: str
    category: ModuleCategory
    
    # Module metadata
    docstring: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    
    # Module contents
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    
    # Metrics
    line_count: int = 0
    complexity_score: float = 0.0
    
    # API surface
    public_functions: List[str] = field(default_factory=list)
    public_classes: List[str] = field(default_factory=list)
    
    # Last analyzed
    last_analyzed: datetime = field(default_factory=datetime.now)


@dataclass
class ModuleRelationship:
    """Relationship between modules."""
    source_module: str
    target_module: str
    relationship_type: RelationshipType
    strength: float  # 0-1, strength of relationship
    
    # Relationship details
    specific_items: List[str] = field(default_factory=list)  # What specifically is related
    line_numbers: List[int] = field(default_factory=list)
    
    # Metadata
    confidence: float = 1.0
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class FunctionalMap:
    """Complete functional map of the codebase."""
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    relationships: List[ModuleRelationship] = field(default_factory=list)
    
    # Graph representation
    dependency_graph: Optional[nx.DiGraph] = None
    
    # Analysis results
    api_surface: Dict[str, List[str]] = field(default_factory=dict)
    business_logic_modules: List[str] = field(default_factory=list)
    core_modules: List[str] = field(default_factory=list)
    utility_modules: List[str] = field(default_factory=list)
    
    # Architecture insights
    architectural_patterns: List[str] = field(default_factory=list)
    design_issues: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class StructureMapper:
    """
    Functional structure mapping system.
    
    Analyzes codebase structure, module relationships, and
    architectural patterns for comprehensive understanding.
    """
    
    @requires_layer("layer3_orchestration", "structure_mapping")
    def __init__(self, watch_paths: Union[str, List[str]]):
        """
        Initialize structure mapper.
        
        Args:
            watch_paths: Directories to analyze
        """
        self.watch_paths = [Path(p) for p in (watch_paths if isinstance(watch_paths, list) else [watch_paths])]
        
        # Analysis cache
        self._functional_map: Optional[FunctionalMap] = None
        self._analysis_cache: Dict[str, Any] = {}
        
        # NEW: Add dashboard integration if enabled
        if FeatureFlags.is_enabled('layer3_orchestration', 'performance_dashboard'):
            self.dashboard = get_performance_dashboard()
            self._setup_dashboard_panels()
        else:
            self.dashboard = None
        
        # Statistics
        self._stats = {
            'modules_analyzed': 0,
            'relationships_found': 0,
            'api_endpoints_identified': 0,
            'business_logic_modules': 0,
            'last_analysis': None
        }
        
        print("🗺️ Structure mapper initialized")
        print(f"   📁 Analyzing: {', '.join(str(p) for p in self.watch_paths)}")
    
    def _setup_dashboard_panels(self):
        """Setup dashboard panels for structure analysis."""
        if not self.dashboard:
            return
        
        from .performance_dashboard import DashboardPanel
        
        # Structure overview panel
        self.dashboard.add_panel(DashboardPanel(
            panel_id="structure_overview",
            title="Codebase Structure Overview",
            panel_type="metric",
            config={
                "metrics": ["total_modules", "relationship_count", "api_endpoints", "complexity_score"],
                "refresh_interval": 10
            }
        ))
        
        # Module category distribution panel
        self.dashboard.add_panel(DashboardPanel(
            panel_id="module_categories",
            title="Module Category Distribution",
            panel_type="chart",
            config={
                "chart_type": "pie",
                "data_source": "module_categories"
            }
        ))
        
        # Architecture insights panel
        self.dashboard.add_panel(DashboardPanel(
            panel_id="architecture_insights",
            title="Architecture Insights",
            panel_type="table",
            config={
                "columns": ["pattern", "status", "details"],
                "max_rows": 20
            }
        ))
        
        # Dependency graph panel
        self.dashboard.add_panel(DashboardPanel(
            panel_id="dependency_graph",
            title="Module Dependencies",
            panel_type="chart",
            config={
                "chart_type": "network",
                "show_weights": True,
                "max_nodes": 50
            }
        ))
    
    def analyze_structure(self, force_reanalysis: bool = False) -> FunctionalMap:
        """
        Analyze complete codebase structure.
        
        Args:
            force_reanalysis: Force re-analysis of all modules
            
        Returns:
            Complete functional map
        """
        print("🔍 Analyzing codebase structure...")
        
        if self._functional_map and not force_reanalysis:
            return self._functional_map
        
        functional_map = FunctionalMap()
        
        # Phase 1: Analyze individual modules
        print("   📊 Phase 1: Analyzing individual modules...")
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
            
            for py_file in watch_path.rglob("*.py"):
                if self._should_analyze_file(py_file):
                    module_info = self._analyze_module(py_file)
                    if module_info:
                        functional_map.modules[str(py_file)] = module_info
                        self._stats['modules_analyzed'] += 1
        
        # Phase 2: Analyze relationships
        print("   🔗 Phase 2: Analyzing module relationships...")
        relationships = self._analyze_relationships(functional_map.modules)
        functional_map.relationships = relationships
        self._stats['relationships_found'] = len(relationships)
        
        # Phase 3: Build dependency graph
        print("   📈 Phase 3: Building dependency graph...")
        functional_map.dependency_graph = self._build_dependency_graph(functional_map)
        
        # Phase 4: Identify architectural elements
        print("   🏗️ Phase 4: Identifying architectural elements...")
        self._identify_architectural_elements(functional_map)
        
        # Phase 5: Detect patterns and issues
        print("   🔍 Phase 5: Detecting patterns and issues...")
        self._detect_patterns_and_issues(functional_map)
        
        functional_map.last_updated = datetime.now()
        self._functional_map = functional_map
        self._stats['last_analysis'] = datetime.now()
        
        print(f"✅ Structure analysis complete: {len(functional_map.modules)} modules, {len(relationships)} relationships")
        
        # Update dashboard with analysis results
        if self.dashboard:
            self._update_dashboard_metrics(functional_map)
        
        return functional_map
    
    def _update_dashboard_metrics(self, functional_map: FunctionalMap):
        """Update dashboard with structure analysis metrics."""
        # Record analysis timing
        record_dashboard_metric("structure_mapper", "analyze_structure", 
                               time.time() - time.time(), MetricType.TIMER)
        
        # Count modules by category
        category_counts = {}
        for category in ModuleCategory:
            category_counts[category.value] = len([
                module for module in functional_map.modules.values()
                if module.category == category
            ])
        
        # Calculate complexity metrics
        total_complexity = sum(module.complexity_score for module in functional_map.modules.values())
        avg_complexity = total_complexity / len(functional_map.modules) if functional_map.modules else 0
        
        # Update structure overview panel
        self.dashboard.update_panel("structure_overview", {
            "total_modules": len(functional_map.modules),
            "relationship_count": len(functional_map.relationships),
            "api_endpoints": len(functional_map.api_surface),
            "complexity_score": avg_complexity,
            "last_updated": datetime.now().isoformat()
        })
        
        # Update module categories panel
        self.dashboard.update_panel("module_categories", {
            "categories": category_counts,
            "total": len(functional_map.modules)
        })
        
        # Update architecture insights panel
        insights_data = []
        for pattern in functional_map.architectural_patterns:
            insights_data.append({
                "pattern": pattern,
                "status": "✅ Detected",
                "details": "Pattern identified in codebase"
            })
        
        for issue in functional_map.design_issues:
            insights_data.append({
                "pattern": "Design Issue",
                "status": "⚠️ Warning", 
                "details": issue
            })
        
        self.dashboard.update_panel("architecture_insights", {
            "insights": insights_data,
            "patterns_found": len(functional_map.architectural_patterns),
            "issues_found": len(functional_map.design_issues)
        })
        
        # Update dependency graph panel
        if functional_map.dependency_graph:
            graph_data = {
                "nodes": len(functional_map.dependency_graph.nodes),
                "edges": len(functional_map.dependency_graph.edges),
                "density": len(functional_map.dependency_graph.edges) / max(len(functional_map.dependency_graph.nodes), 1),
                "core_modules": len(functional_map.core_modules)
            }
            self.dashboard.update_panel("dependency_graph", graph_data)
        
        # Record component performance metrics
        record_dashboard_metric("structure_mapper", "modules_analyzed", 
                               len(functional_map.modules), MetricType.GAUGE)
        record_dashboard_metric("structure_mapper", "relationships_found", 
                               len(functional_map.relationships), MetricType.GAUGE)
        record_dashboard_metric("structure_mapper", "average_complexity", 
                               avg_complexity, MetricType.GAUGE)
    
    def _analyze_module(self, file_path: Path) -> Optional[ModuleInfo]:
        """Analyze a single module."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract module information
            module_info = ModuleInfo(
                module_path=str(file_path),
                category=self._categorize_module(file_path, tree),
                docstring=ast.get_docstring(tree),
                line_count=len(content.split('\\n'))
            )
            
            # Analyze imports
            module_info.imports = self._extract_imports(tree)
            
            # Analyze functions
            module_info.functions = self._extract_functions(tree, str(file_path))
            
            # Analyze classes
            module_info.classes = self._extract_classes(tree, str(file_path))
            
            # Analyze constants
            module_info.constants = self._extract_constants(tree)
            
            # Identify API surface
            self._identify_api_surface(module_info)
            
            # Calculate complexity
            module_info.complexity_score = self._calculate_module_complexity(module_info)
            
            return module_info
            
        except Exception as e:
            print(f"⚠️ Error analyzing module {file_path}: {e}")
            return None
    
    def _categorize_module(self, file_path: Path, tree: ast.AST) -> ModuleCategory:
        """Categorize module based on its content and path."""
        path_str = str(file_path).lower()
        name = file_path.name.lower()
        
        # Test modules
        if (name.startswith('test_') or name.endswith('_test.py') or 
            'test' in str(file_path.parent).lower()):
            return ModuleCategory.TESTING
        
        # API modules
        if any(keyword in path_str for keyword in ['api', 'endpoint', 'route', 'handler', 'controller']):
            return ModuleCategory.API_LAYER
        
        # Data access modules
        if any(keyword in path_str for keyword in ['model', 'dao', 'repository', 'database', 'db']):
            return ModuleCategory.DATA_ACCESS
        
        # UI modules
        if any(keyword in path_str for keyword in ['ui', 'view', 'template', 'frontend', 'gui']):
            return ModuleCategory.USER_INTERFACE
        
        # Configuration modules
        if any(keyword in path_str for keyword in ['config', 'setting', 'env']):
            return ModuleCategory.CONFIGURATION
        
        # Utility modules
        if any(keyword in path_str for keyword in ['util', 'helper', 'common', 'shared', 'tool']):
            return ModuleCategory.UTILITY
        
        # Infrastructure modules
        if any(keyword in path_str for keyword in ['infra', 'service', 'client', 'driver']):
            return ModuleCategory.INFRASTRUCTURE
        
        # External integration
        if any(keyword in path_str for keyword in ['integration', 'external', 'third_party']):
            return ModuleCategory.EXTERNAL_INTEGRATION
        
        # Check content for business logic indicators
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Look for business domain terms in function/class names
        business_terms = ['process', 'calculate', 'validate', 'analyze', 'manage', 'execute', 'handle']
        
        has_business_logic = False
        for func in functions:
            if any(term in func.name.lower() for term in business_terms):
                has_business_logic = True
                break
        
        for cls in classes:
            if any(term in cls.name.lower() for term in business_terms):
                has_business_logic = True
                break
        
        if has_business_logic:
            return ModuleCategory.CORE_BUSINESS
        
        return ModuleCategory.UTILITY
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    
    def _extract_functions(self, tree: ast.AST, module_path: str) -> List[FunctionInfo]:
        """Extract function information from AST."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip nested functions (methods will be handled with classes)
                if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                      if any(child for child in ast.iter_child_nodes(parent) if child is node)):
                    continue
                
                func_info = FunctionInfo(
                    name=node.name,
                    module_path=module_path,
                    line_number=node.lineno,
                    access_level=self._determine_access_level(node.name),
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    docstring=ast.get_docstring(node),
                    parameters=[arg.arg for arg in node.args.args],
                    decorators=[self._get_decorator_name(dec) for dec in node.decorator_list],
                    line_count=self._count_node_lines(node),
                    cyclomatic_complexity=self._calculate_cyclomatic_complexity(node)
                )
                
                # Extract return annotation
                if node.returns:
                    func_info.return_annotation = ast.unparse(node.returns)
                
                # Extract function calls
                func_info.calls_functions = self._extract_function_calls(node)
                
                functions.append(func_info)
        
        return functions
    
    def _extract_classes(self, tree: ast.AST, module_path: str) -> List[ClassInfo]:
        """Extract class information from AST."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Skip nested classes for now
                if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                      if any(child for child in ast.iter_child_nodes(parent) if child is node)):
                    continue
                
                class_info = ClassInfo(
                    name=node.name,
                    module_path=module_path,
                    line_number=node.lineno,
                    access_level=self._determine_access_level(node.name),
                    docstring=ast.get_docstring(node),
                    base_classes=[ast.unparse(base) for base in node.bases],
                    decorators=[self._get_decorator_name(dec) for dec in node.decorator_list],
                    line_count=self._count_node_lines(node)
                )
                
                # Check class characteristics
                class_info.is_dataclass = any(
                    isinstance(dec, ast.Name) and dec.id == 'dataclass'
                    for dec in node.decorator_list
                )
                
                class_info.is_enum = any(
                    'Enum' in ast.unparse(base) for base in node.bases
                )
                
                class_info.is_exception = any(
                    'Exception' in ast.unparse(base) or 'Error' in ast.unparse(base)
                    for base in node.bases
                )
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_info = FunctionInfo(
                            name=item.name,
                            module_path=module_path,
                            line_number=item.lineno,
                            access_level=self._determine_access_level(item.name),
                            is_async=isinstance(item, ast.AsyncFunctionDef),
                            is_method=True,
                            docstring=ast.get_docstring(item),
                            parameters=[arg.arg for arg in item.args.args],
                            decorators=[self._get_decorator_name(dec) for dec in item.decorator_list],
                            line_count=self._count_node_lines(item),
                            cyclomatic_complexity=self._calculate_cyclomatic_complexity(item)
                        )
                        
                        # Check for special method types
                        if any(dec.id == 'staticmethod' for dec in item.decorator_list 
                              if isinstance(dec, ast.Name)):
                            method_info.is_staticmethod = True
                        
                        if any(dec.id == 'classmethod' for dec in item.decorator_list 
                              if isinstance(dec, ast.Name)):
                            method_info.is_classmethod = True
                        
                        if any(dec.id == 'property' for dec in item.decorator_list 
                              if isinstance(dec, ast.Name)):
                            method_info.is_property = True
                            class_info.properties.append(item.name)
                        
                        class_info.methods.append(method_info)
                
                class_info.method_count = len(class_info.methods)
                classes.append(class_info)
        
        return classes
    
    def _extract_constants(self, tree: ast.AST) -> List[str]:
        """Extract module-level constants."""
        constants = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(target.id)
        
        return constants
    
    def _determine_access_level(self, name: str) -> AccessLevel:
        """Determine access level from name."""
        if name.startswith('__') and name.endswith('__'):
            return AccessLevel.PUBLIC  # Special methods are public
        elif name.startswith('__'):
            return AccessLevel.PRIVATE
        elif name.startswith('_'):
            return AccessLevel.PROTECTED
        else:
            return AccessLevel.PUBLIC
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name from AST node."""
        try:
            return ast.unparse(decorator)
        except:
            return "unknown_decorator"
    
    def _count_node_lines(self, node: ast.AST) -> int:
        """Count lines in an AST node."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - node.lineno + 1
        return 1
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls from a function."""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    try:
                        calls.append(ast.unparse(child.func))
                    except:
                        pass
        
        return calls
    
    def _identify_api_surface(self, module_info: ModuleInfo):
        """Identify public API surface of a module."""
        # Public functions
        module_info.public_functions = [
            func.name for func in module_info.functions
            if func.access_level == AccessLevel.PUBLIC
        ]
        
        # Public classes
        module_info.public_classes = [
            cls.name for cls in module_info.classes
            if cls.access_level == AccessLevel.PUBLIC
        ]
        
        # Update exports (simplified)
        module_info.exports = module_info.public_functions + module_info.public_classes
    
    def _calculate_module_complexity(self, module_info: ModuleInfo) -> float:
        """Calculate overall module complexity score."""
        complexity = 0.0
        
        # Line count contribution
        complexity += module_info.line_count / 20  # 1 point per 20 lines
        
        # Function complexity
        for func in module_info.functions:
            complexity += func.cyclomatic_complexity * 0.5
        
        # Class complexity
        for cls in module_info.classes:
            complexity += len(cls.methods) * 0.3
            complexity += sum(method.cyclomatic_complexity for method in cls.methods) * 0.2
        
        # Import complexity
        complexity += len(module_info.imports) * 0.1
        
        return min(complexity, 100.0)  # Cap at 100
    
    def _analyze_relationships(self, modules: Dict[str, ModuleInfo]) -> List[ModuleRelationship]:
        """Analyze relationships between modules."""
        relationships = []
        
        for source_path, source_module in modules.items():
            source_name = Path(source_path).stem
            
            # Analyze imports
            for import_name in source_module.imports:
                # Find matching modules
                for target_path, target_module in modules.items():
                    target_name = Path(target_path).stem
                    
                    if (import_name == target_name or 
                        import_name.endswith(target_name) or
                        target_name in import_name):
                        
                        relationship = ModuleRelationship(
                            source_module=source_path,
                            target_module=target_path,
                            relationship_type=RelationshipType.IMPORTS,
                            strength=0.7,
                            specific_items=[import_name]
                        )
                        relationships.append(relationship)
            
            # Analyze inheritance relationships
            for cls in source_module.classes:
                for base_class in cls.base_classes:
                    # Find modules that define this base class
                    for target_path, target_module in modules.items():
                        if any(target_cls.name in base_class for target_cls in target_module.classes):
                            relationship = ModuleRelationship(
                                source_module=source_path,
                                target_module=target_path,
                                relationship_type=RelationshipType.INHERITS,
                                strength=0.9,
                                specific_items=[base_class],
                                line_numbers=[cls.line_number]
                            )
                            relationships.append(relationship)
            
            # Analyze function calls
            all_calls = []
            for func in source_module.functions:
                all_calls.extend(func.calls_functions)
            for cls in source_module.classes:
                for method in cls.methods:
                    all_calls.extend(method.calls_functions)
            
            # Map calls to modules
            call_counts = {}
            for call in all_calls:
                for target_path, target_module in modules.items():
                    if target_path != source_path:  # Don't self-reference
                        target_functions = [f.name for f in target_module.functions]
                        target_methods = []
                        for cls in target_module.classes:
                            target_methods.extend([m.name for m in cls.methods])
                        
                        if call in target_functions or call in target_methods:
                            call_counts[target_path] = call_counts.get(target_path, 0) + 1
            
            # Create call relationships
            for target_path, call_count in call_counts.items():
                strength = min(call_count / 10, 1.0)  # Normalize to 0-1
                relationship = ModuleRelationship(
                    source_module=source_path,
                    target_module=target_path,
                    relationship_type=RelationshipType.CALLS,
                    strength=strength,
                    specific_items=[f"{call_count} function calls"]
                )
                relationships.append(relationship)
        
        return relationships
    
    def _build_dependency_graph(self, functional_map: FunctionalMap) -> nx.DiGraph:
        """Build dependency graph from relationships."""
        graph = nx.DiGraph()
        
        # Add nodes
        for module_path in functional_map.modules.keys():
            module_name = Path(module_path).stem
            graph.add_node(module_name, path=module_path)
        
        # Add edges
        for relationship in functional_map.relationships:
            source_name = Path(relationship.source_module).stem
            target_name = Path(relationship.target_module).stem
            
            if graph.has_edge(source_name, target_name):
                # Update existing edge
                graph[source_name][target_name]['weight'] += relationship.strength
                graph[source_name][target_name]['types'].append(relationship.relationship_type.value)
            else:
                # Add new edge
                graph.add_edge(
                    source_name,
                    target_name,
                    weight=relationship.strength,
                    types=[relationship.relationship_type.value]
                )
        
        return graph
    
    def _identify_architectural_elements(self, functional_map: FunctionalMap):
        """Identify key architectural elements."""
        # Identify API surface
        for module_path, module_info in functional_map.modules.items():
            if module_info.category == ModuleCategory.API_LAYER:
                functional_map.api_surface[module_path] = module_info.exports
        
        # Identify business logic modules
        functional_map.business_logic_modules = [
            module_path for module_path, module_info in functional_map.modules.items()
            if module_info.category == ModuleCategory.CORE_BUSINESS
        ]
        
        # Identify core modules (highly connected)
        if functional_map.dependency_graph:
            # Calculate centrality
            centrality = nx.degree_centrality(functional_map.dependency_graph)
            
            # Top 20% most connected modules are considered core
            sorted_modules = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            top_20_percent = max(1, len(sorted_modules) // 5)
            
            core_module_names = [name for name, _ in sorted_modules[:top_20_percent]]
            functional_map.core_modules = [
                module_path for module_path in functional_map.modules.keys()
                if Path(module_path).stem in core_module_names
            ]
        
        # Identify utility modules
        functional_map.utility_modules = [
            module_path for module_path, module_info in functional_map.modules.items()
            if module_info.category == ModuleCategory.UTILITY
        ]
    
    def _detect_patterns_and_issues(self, functional_map: FunctionalMap):
        """Detect architectural patterns and potential issues."""
        patterns = []
        issues = []
        
        # Detect patterns
        if functional_map.dependency_graph:
            # Check for layered architecture
            api_modules = [Path(p).stem for p in functional_map.modules 
                          if functional_map.modules[p].category == ModuleCategory.API_LAYER]
            business_modules = [Path(p).stem for p in functional_map.modules
                               if functional_map.modules[p].category == ModuleCategory.CORE_BUSINESS]
            data_modules = [Path(p).stem for p in functional_map.modules
                           if functional_map.modules[p].category == ModuleCategory.DATA_ACCESS]
            
            if api_modules and business_modules and data_modules:
                patterns.append("Layered Architecture (API -> Business -> Data)")
            
            # Check for circular dependencies
            try:
                cycles = list(nx.simple_cycles(functional_map.dependency_graph))
                if cycles:
                    issues.append(f"Circular dependencies detected: {len(cycles)} cycles")
                else:
                    patterns.append("Acyclic dependency structure")
            except:
                pass
            
            # Check for God modules (too many connections)
            centrality = nx.degree_centrality(functional_map.dependency_graph)
            god_modules = [name for name, cent in centrality.items() if cent > 0.5]
            if god_modules:
                issues.append(f"Potential God modules: {', '.join(god_modules)}")
            
            # Check for orphaned modules (no connections)
            isolated = list(nx.isolates(functional_map.dependency_graph))
            if isolated:
                issues.append(f"Isolated modules: {', '.join(isolated)}")
        
        # Detect complexity issues
        high_complexity_modules = [
            Path(path).stem for path, module in functional_map.modules.items()
            if module.complexity_score > 80
        ]
        if high_complexity_modules:
            issues.append(f"High complexity modules: {', '.join(high_complexity_modules)}")
        
        # Detect missing documentation
        undocumented_modules = [
            Path(path).stem for path, module in functional_map.modules.items()
            if not module.docstring and module.category in [
                ModuleCategory.CORE_BUSINESS, ModuleCategory.API_LAYER
            ]
        ]
        if undocumented_modules:
            issues.append(f"Undocumented critical modules: {', '.join(undocumented_modules)}")
        
        functional_map.architectural_patterns = patterns
        functional_map.design_issues = issues
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed."""
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
    
    def get_module_dependencies(self, module_path: str) -> List[str]:
        """Get dependencies of a specific module."""
        if not self._functional_map:
            return []
        
        dependencies = []
        for relationship in self._functional_map.relationships:
            if relationship.source_module == module_path:
                dependencies.append(relationship.target_module)
        
        return dependencies
    
    def get_module_dependents(self, module_path: str) -> List[str]:
        """Get modules that depend on a specific module."""
        if not self._functional_map:
            return []
        
        dependents = []
        for relationship in self._functional_map.relationships:
            if relationship.target_module == module_path:
                dependents.append(relationship.source_module)
        
        return dependents
    
    def get_critical_modules(self) -> List[str]:
        """Get modules that are critical to the system."""
        if not self._functional_map or not self._functional_map.dependency_graph:
            return []
        
        # Calculate betweenness centrality (modules that are bridges)
        centrality = nx.betweenness_centrality(self._functional_map.dependency_graph)
        
        # Top 10% most critical modules
        sorted_modules = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_10_percent = max(1, len(sorted_modules) // 10)
        
        critical_names = [name for name, _ in sorted_modules[:top_10_percent]]
        
        # Map back to full paths
        critical_modules = []
        for module_path in self._functional_map.modules.keys():
            if Path(module_path).stem in critical_names:
                critical_modules.append(module_path)
        
        return critical_modules
    
    def get_structure_statistics(self) -> Dict[str, Any]:
        """Get structure analysis statistics."""
        if not self._functional_map:
            return {"error": "No analysis performed yet"}
        
        # Count modules by category
        category_counts = {}
        for category in ModuleCategory:
            category_counts[category.value] = len([
                module for module in self._functional_map.modules.values()
                if module.category == category
            ])
        
        # Graph metrics
        graph_metrics = {}
        if self._functional_map.dependency_graph:
            graph_metrics = {
                "nodes": len(self._functional_map.dependency_graph.nodes),
                "edges": len(self._functional_map.dependency_graph.edges),
                "density": nx.density(self._functional_map.dependency_graph),
                "is_connected": nx.is_weakly_connected(self._functional_map.dependency_graph)
            }
        
        return {
            "total_modules": len(self._functional_map.modules),
            "total_relationships": len(self._functional_map.relationships),
            "category_distribution": category_counts,
            "api_endpoints": len(self._functional_map.api_surface),
            "business_logic_modules": len(self._functional_map.business_logic_modules),
            "core_modules": len(self._functional_map.core_modules),
            "utility_modules": len(self._functional_map.utility_modules),
            "architectural_patterns": len(self._functional_map.architectural_patterns),
            "design_issues": len(self._functional_map.design_issues),
            "graph_metrics": graph_metrics,
            "statistics": dict(self._stats)
        }
    
    def export_structure_report(self, output_path: str = "structure_report.json"):
        """Export comprehensive structure report."""
        if not self._functional_map:
            print("⚠️ No analysis performed yet. Run analyze_structure() first.")
            return
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "analysis_timestamp": self._functional_map.last_updated.isoformat(),
            "statistics": self.get_structure_statistics(),
            "modules": {},
            "relationships": [],
            "architectural_insights": {
                "patterns": self._functional_map.architectural_patterns,
                "issues": self._functional_map.design_issues,
                "api_surface": self._functional_map.api_surface,
                "core_modules": self._functional_map.core_modules,
                "business_logic_modules": self._functional_map.business_logic_modules
            }
        }
        
        # Add module details
        for module_path, module_info in self._functional_map.modules.items():
            module_data = {
                "category": module_info.category.value,
                "line_count": module_info.line_count,
                "complexity_score": module_info.complexity_score,
                "function_count": len(module_info.functions),
                "class_count": len(module_info.classes),
                "public_api_count": len(module_info.exports),
                "import_count": len(module_info.imports),
                "last_analyzed": module_info.last_analyzed.isoformat()
            }
            report["modules"][module_path] = module_data
        
        # Add relationship details
        for relationship in self._functional_map.relationships:
            rel_data = {
                "source": relationship.source_module,
                "target": relationship.target_module,
                "type": relationship.relationship_type.value,
                "strength": relationship.strength,
                "items": relationship.specific_items
            }
            report["relationships"].append(rel_data)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Structure report exported to {output_path}")
        except Exception as e:
            print(f"⚠️ Error exporting structure report: {e}")


# Convenience functions for structure analysis
def analyze_directory_structure(directory: str) -> FunctionalMap:
    """Quick structure analysis of a directory."""
    mapper = StructureMapper(directory)
    return mapper.analyze_structure()


def find_critical_modules(directory: str) -> List[str]:
    """Find critical modules in a directory."""
    mapper = StructureMapper(directory)
    mapper.analyze_structure()
    return mapper.get_critical_modules()