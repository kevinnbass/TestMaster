"""
Module Dependency Tracker

Inspired by LangGraph's graph-based workflow patterns for tracking
module dependencies and impact analysis.

Features:
- Dependency graph construction
- Impact analysis for changes
- Cascade testing recommendations
- Topological sorting for build order
"""

import ast
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque

from ..core.layer_manager import requires_layer


@dataclass
class ModuleDependency:
    """Dependency relationship between modules."""
    source_module: str
    target_module: str
    dependency_type: str  # "import", "function_call", "inheritance", "type_hint"
    strength: float  # 0.0 to 1.0 - how critical the dependency is
    line_numbers: List[int] = field(default_factory=list)
    import_statements: List[str] = field(default_factory=list)
    last_verified: datetime = field(default_factory=datetime.now)


@dataclass
class ImpactAnalysis:
    """Analysis of what modules are impacted by a change."""
    changed_module: str
    directly_impacted: List[str]
    indirectly_impacted: List[str]
    tests_to_run: List[str]
    risk_score: float  # 0.0 to 1.0
    estimated_cascade_size: int


class DependencyTracker:
    """
    Track module dependencies and perform impact analysis.
    
    Uses graph-based approach similar to LangGraph for dependency management
    and traversal algorithms for impact analysis.
    """
    
    @requires_layer("layer1_test_foundation", "test_mapping")
    def __init__(self, source_dir: Union[str, Path]):
        """
        Initialize dependency tracker.
        
        Args:
            source_dir: Directory containing source code
        """
        self.source_dir = Path(source_dir)
        
        # Dependency graph (NetworkX for graph algorithms)
        self.dependency_graph = nx.DiGraph()
        
        # Dependency storage
        self.dependencies: Dict[str, List[ModuleDependency]] = defaultdict(list)
        
        # Cache for expensive operations
        self._ast_cache = {}
        self._import_cache = {}
        
    def build_dependency_graph(self) -> nx.DiGraph:
        """
        Build complete dependency graph for all modules.
        
        Returns:
            NetworkX directed graph with modules as nodes and dependencies as edges
        """
        print("🕸️ Building module dependency graph...")
        
        # Clear existing graph
        self.dependency_graph.clear()
        self.dependencies.clear()
        
        # Discover all modules
        modules = self._discover_modules()
        print(f"📁 Found {len(modules)} modules to analyze")
        
        # Add modules as nodes
        for module in modules:
            module_name = self._get_module_name(module)
            self.dependency_graph.add_node(module_name, file_path=str(module))
        
        # Analyze dependencies for each module
        for module in modules:
            self._analyze_module_dependencies(module)
        
        # Add edges to graph
        self._build_graph_edges()
        
        self._print_dependency_summary()
        return self.dependency_graph
    
    def _discover_modules(self) -> List[Path]:
        """Discover all Python modules in source directory."""
        modules = []
        
        for py_file in self.source_dir.rglob("*.py"):
            if self._should_analyze_module(py_file):
                modules.append(py_file)
        
        return sorted(modules)
    
    def _should_analyze_module(self, file_path: Path) -> bool:
        """Check if module should be analyzed for dependencies."""
        if "__pycache__" in str(file_path):
            return False
        if file_path.name.startswith("_") and file_path.stem != "__init__":
            return False
        if "test" in file_path.name.lower():
            return False
        return True
    
    def _get_module_name(self, file_path: Path) -> str:
        """Get standardized module name from file path."""
        relative_path = file_path.relative_to(self.source_dir)
        if relative_path.name == "__init__.py":
            return str(relative_path.parent).replace('\\', '.').replace('/', '.')
        else:
            return str(relative_path.with_suffix('')).replace('\\', '.').replace('/', '.')
    
    def _analyze_module_dependencies(self, module_path: Path):
        """Analyze dependencies for a single module."""
        try:
            module_name = self._get_module_name(module_path)
            
            # Parse AST
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            # Extract different types of dependencies
            dependencies = []
            
            # Import dependencies
            dependencies.extend(self._extract_import_dependencies(tree, module_name, content))
            
            # Function call dependencies
            dependencies.extend(self._extract_function_call_dependencies(tree, module_name))
            
            # Inheritance dependencies
            dependencies.extend(self._extract_inheritance_dependencies(tree, module_name))
            
            # Type hint dependencies
            dependencies.extend(self._extract_type_hint_dependencies(tree, module_name))
            
            # Store dependencies
            self.dependencies[module_name] = dependencies
            
        except Exception as e:
            print(f"⚠️ Error analyzing dependencies for {module_path}: {e}")
    
    def _extract_import_dependencies(self, tree: ast.AST, module_name: str, 
                                   content: str) -> List[ModuleDependency]:
        """Extract import-based dependencies."""
        dependencies = []
        lines = content.split('\\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    if self._is_internal_module(target):
                        dependency = ModuleDependency(
                            source_module=module_name,
                            target_module=target,
                            dependency_type="import",
                            strength=0.8,  # Direct imports are strong dependencies
                            line_numbers=[node.lineno],
                            import_statements=[f"import {target}"]
                        )
                        dependencies.append(dependency)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and self._is_internal_module(node.module):
                    # Calculate strength based on what's imported
                    imported_items = [alias.name for alias in node.names]
                    strength = 0.9 if '*' in imported_items else 0.7
                    
                    dependency = ModuleDependency(
                        source_module=module_name,
                        target_module=node.module,
                        dependency_type="import",
                        strength=strength,
                        line_numbers=[node.lineno],
                        import_statements=[f"from {node.module} import {', '.join(imported_items)}"]
                    )
                    dependencies.append(dependency)
        
        return dependencies
    
    def _extract_function_call_dependencies(self, tree: ast.AST, 
                                          module_name: str) -> List[ModuleDependency]:
        """Extract function call dependencies."""
        dependencies = []
        
        # This would require more sophisticated analysis to track
        # cross-module function calls. For now, we'll focus on imports.
        # Could be enhanced to track:
        # - module.function() calls
        # - Dynamic imports
        # - Callback registrations
        
        return dependencies
    
    def _extract_inheritance_dependencies(self, tree: ast.AST, 
                                        module_name: str) -> List[ModuleDependency]:
        """Extract inheritance-based dependencies."""
        dependencies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Attribute):
                        # Handle module.Class inheritance
                        if isinstance(base.value, ast.Name):
                            target_module = base.value.id
                            if self._is_internal_module(target_module):
                                dependency = ModuleDependency(
                                    source_module=module_name,
                                    target_module=target_module,
                                    dependency_type="inheritance",
                                    strength=0.9,  # Inheritance is a strong dependency
                                    line_numbers=[node.lineno]
                                )
                                dependencies.append(dependency)
        
        return dependencies
    
    def _extract_type_hint_dependencies(self, tree: ast.AST, 
                                      module_name: str) -> List[ModuleDependency]:
        """Extract type hint dependencies."""
        dependencies = []
        
        # This would analyze type annotations for cross-module type references
        # Could track:
        # - Function parameter types
        # - Return type annotations
        # - Variable type hints
        # - Generic type parameters
        
        return dependencies
    
    def _is_internal_module(self, module_name: str) -> bool:
        """Check if module is internal to the project."""
        # Simple heuristic - internal modules don't contain dots or are relative
        if not module_name:
            return False
        
        # Skip standard library and third-party modules
        stdlib_modules = {'os', 'sys', 'json', 'ast', 'pathlib', 'datetime', 'typing'}
        if module_name.split('.')[0] in stdlib_modules:
            return False
        
        # Skip common third-party packages
        third_party = {'numpy', 'pandas', 'requests', 'flask', 'django', 'pytest'}
        if module_name.split('.')[0] in third_party:
            return False
        
        return True
    
    def _build_graph_edges(self):
        """Add edges to dependency graph based on discovered dependencies."""
        for source_module, deps in self.dependencies.items():
            for dep in deps:
                if self.dependency_graph.has_node(dep.target_module):
                    self.dependency_graph.add_edge(
                        source_module, 
                        dep.target_module,
                        dependency_type=dep.dependency_type,
                        strength=dep.strength,
                        line_numbers=dep.line_numbers
                    )
    
    def analyze_impact(self, changed_modules: Union[str, List[str]]) -> ImpactAnalysis:
        """
        Analyze impact of changes to specific modules.
        
        Args:
            changed_modules: Module(s) that changed
            
        Returns:
            Impact analysis with affected modules and recommendations
        """
        if isinstance(changed_modules, str):
            changed_modules = [changed_modules]
        
        print(f"📊 Analyzing impact of changes to: {changed_modules}")
        
        all_impacted = set()
        directly_impacted = set()
        
        # Find all modules that depend on the changed modules
        for changed_module in changed_modules:
            if changed_module in self.dependency_graph:
                # Direct dependents (modules that import this one)
                direct_deps = set(self.dependency_graph.predecessors(changed_module))
                directly_impacted.update(direct_deps)
                
                # Indirect dependents (transitive closure)
                all_deps = set()
                visited = set()
                queue = deque([changed_module])
                
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    predecessors = list(self.dependency_graph.predecessors(current))
                    all_deps.update(predecessors)
                    queue.extend(predecessors)
                
                all_impacted.update(all_deps)
        
        indirectly_impacted = all_impacted - directly_impacted
        
        # Calculate risk score based on impact size and dependency strength
        risk_score = min(1.0, len(all_impacted) / max(len(self.dependency_graph.nodes), 1))
        
        # Estimate cascade size
        cascade_size = len(all_impacted)
        
        # Recommend tests to run (would integrate with test mapper)
        tests_to_run = list(all_impacted)  # Simplified - would map to actual tests
        
        analysis = ImpactAnalysis(
            changed_module=", ".join(changed_modules),
            directly_impacted=list(directly_impacted),
            indirectly_impacted=list(indirectly_impacted),
            tests_to_run=tests_to_run,
            risk_score=risk_score,
            estimated_cascade_size=cascade_size
        )
        
        self._print_impact_analysis(analysis)
        return analysis
    
    def get_build_order(self) -> List[str]:
        """
        Get topological build order for modules.
        
        Returns:
            List of modules in dependency order (dependencies first)
        """
        try:
            # Topological sort gives us dependency order
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXError:
            print("⚠️ Circular dependencies detected, cannot determine build order")
            return list(self.dependency_graph.nodes)
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the module graph.
        
        Returns:
            List of cycles (each cycle is a list of module names)
        """
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                print(f"⚠️ Found {len(cycles)} circular dependencies")
                for i, cycle in enumerate(cycles):
                    print(f"   Cycle {i+1}: {' -> '.join(cycle + [cycle[0]])}")
            return cycles
        except Exception as e:
            print(f"⚠️ Error detecting cycles: {e}")
            return []
    
    def get_critical_modules(self, threshold: float = 0.8) -> List[str]:
        """
        Get modules that many others depend on (critical modules).
        
        Args:
            threshold: Minimum dependency ratio to be considered critical
            
        Returns:
            List of critical module names
        """
        critical_modules = []
        total_modules = len(self.dependency_graph.nodes)
        
        for node in self.dependency_graph.nodes:
            dependents = list(self.dependency_graph.predecessors(node))
            dependency_ratio = len(dependents) / max(total_modules - 1, 1)
            
            if dependency_ratio >= threshold:
                critical_modules.append(node)
        
        return sorted(critical_modules, key=lambda m: len(list(self.dependency_graph.predecessors(m))), reverse=True)
    
    def get_isolated_modules(self) -> List[str]:
        """Get modules with no dependencies (leaf modules)."""
        return [
            node for node in self.dependency_graph.nodes
            if self.dependency_graph.in_degree(node) == 0 and self.dependency_graph.out_degree(node) == 0
        ]
    
    def _print_dependency_summary(self):
        """Print summary of dependency analysis."""
        print("\\n" + "="*60)
        print("🕸️ DEPENDENCY ANALYSIS SUMMARY")
        print("="*60)
        
        total_modules = len(self.dependency_graph.nodes)
        total_edges = len(self.dependency_graph.edges)
        
        print(f"📊 Total Modules: {total_modules}")
        print(f"🔗 Total Dependencies: {total_edges}")
        
        if total_modules > 0:
            avg_deps = total_edges / total_modules
            print(f"📈 Average Dependencies per Module: {avg_deps:.1f}")
        
        # Analyze dependency types
        dependency_types = defaultdict(int)
        for _, _, data in self.dependency_graph.edges(data=True):
            dependency_types[data.get('dependency_type', 'unknown')] += 1
        
        if dependency_types:
            print(f"\\n🔍 Dependency Types:")
            for dep_type, count in dependency_types.items():
                print(f"   {dep_type.title()}: {count}")
        
        # Check for circular dependencies
        cycles = self.detect_circular_dependencies()
        if not cycles:
            print(f"\\n✅ No circular dependencies detected")
        
        # Show critical modules
        critical = self.get_critical_modules(threshold=0.3)
        if critical:
            print(f"\\n⚡ Critical Modules (many dependents):")
            for module in critical[:5]:  # Top 5
                dependents = len(list(self.dependency_graph.predecessors(module)))
                print(f"   {module}: {dependents} dependents")
        
        print("="*60)
    
    def _print_impact_analysis(self, analysis: ImpactAnalysis):
        """Print impact analysis results."""
        print("\\n" + "="*60)
        print("📊 IMPACT ANALYSIS RESULTS")
        print("="*60)
        
        print(f"🎯 Changed Module(s): {analysis.changed_module}")
        print(f"⚠️ Risk Score: {analysis.risk_score:.2f}")
        print(f"📏 Cascade Size: {analysis.estimated_cascade_size}")
        
        print(f"\\n🔍 Direct Impact ({len(analysis.directly_impacted)} modules):")
        for module in analysis.directly_impacted[:10]:  # Show first 10
            print(f"   📦 {module}")
        
        if len(analysis.indirectly_impacted) > 0:
            print(f"\\n🌊 Indirect Impact ({len(analysis.indirectly_impacted)} modules):")
            for module in analysis.indirectly_impacted[:10]:  # Show first 10
                print(f"   📦 {module}")
        
        print(f"\\n🧪 Recommended Tests ({len(analysis.tests_to_run)}):")
        for test in analysis.tests_to_run[:10]:  # Show first 10
            print(f"   🔬 {test}")
        
        print("="*60)