#!/usr/bin/env python3
"""
TestMaster Dependency Graph Analysis - Agent B Phase 2
======================================================

Comprehensive dependency analysis and graph construction for the TestMaster
intelligence framework. Creates detailed dependency mappings for Neo4j export
and architectural understanding.

Author: Agent B - Documentation & Modularization Excellence
Phase: 2 - Advanced Interdependency Analysis (Hours 26-30)
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import networkx as nx


@dataclass
class ModuleNode:
    """Represents a module in the dependency graph."""
    id: str
    name: str
    path: str
    import_count: int = 0
    export_count: int = 0
    complexity: float = 0.0
    lines_of_code: int = 0
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    module_type: str = "standard"  # standard, test, config, api, intelligence
    documentation_score: float = 0.0
    coupling_score: float = 0.0
    cohesion_score: float = 0.0


@dataclass
class DependencyEdge:
    """Represents a dependency relationship between modules."""
    source: str
    target: str
    import_type: str  # direct, relative, star, specific
    import_count: int = 1
    coupling_strength: float = 0.0
    dependency_type: str = "standard"  # standard, circular, critical


class DependencyGraphAnalyzer:
    """
    Comprehensive dependency graph analyzer for TestMaster framework.
    
    Analyzes module relationships, coupling, and generates Neo4j-compatible
    dependency graphs for architectural understanding and optimization.
    """
    
    def __init__(self, root_path: str = "."):
        """Initialize dependency graph analyzer.
        
        Args:
            root_path: Root path of the TestMaster framework
        """
        self.root_path = Path(root_path)
        self.modules: Dict[str, ModuleNode] = {}
        self.edges: List[DependencyEdge] = []
        self.graph = nx.DiGraph()
        self.analysis_results = {}
        
        # Critical intelligence modules to prioritize
        self.intelligence_modules = {
            "core/intelligence/__init__.py",
            "core/intelligence/testing/__init__.py", 
            "core/intelligence/api/__init__.py",
            "core/intelligence/analytics/__init__.py",
            "testmaster_orchestrator.py",
            "intelligent_test_builder.py",
            "enhanced_self_healing_verifier.py",
            "agentic_test_monitor.py",
            "parallel_converter.py"
        }
    
    def analyze_module_dependencies(self) -> Dict[str, Any]:
        """
        Perform comprehensive dependency analysis.
        
        Returns:
            Complete analysis results including graph metrics and insights
        """
        print("🔍 Starting comprehensive dependency analysis...")
        
        # Step 1: Discover and analyze all Python modules
        self._discover_modules()
        
        # Step 2: Analyze import relationships
        self._analyze_imports()
        
        # Step 3: Build NetworkX graph for analysis
        self._build_networkx_graph()
        
        # Step 4: Calculate coupling and cohesion metrics
        self._calculate_coupling_metrics()
        
        # Step 5: Detect circular dependencies
        circular_deps = self._detect_circular_dependencies()
        
        # Step 6: Identify critical modules
        critical_modules = self._identify_critical_modules()
        
        # Step 7: Calculate graph metrics
        graph_metrics = self._calculate_graph_metrics()
        
        # Step 8: Generate insights
        insights = self._generate_insights()
        
        self.analysis_results = {
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'analyzer': 'Agent B - Dependency Graph Analysis',
                'total_modules': len(self.modules),
                'total_dependencies': len(self.edges),
                'intelligence_modules': len(self.intelligence_modules)
            },
            'modules': {id: asdict(module) for id, module in self.modules.items()},
            'dependencies': [asdict(edge) for edge in self.edges],
            'circular_dependencies': circular_deps,
            'critical_modules': critical_modules,
            'graph_metrics': graph_metrics,
            'insights': insights,
            'neo4j_export': self._prepare_neo4j_export()
        }
        
        print(f"✅ Analysis complete: {len(self.modules)} modules, {len(self.edges)} dependencies")
        return self.analysis_results
    
    def _discover_modules(self):
        """Discover all Python modules in the framework."""
        print("📁 Discovering Python modules...")
        
        for py_file in self.root_path.rglob("*.py"):
            if self._should_analyze_file(py_file):
                module_id = self._get_module_id(py_file)
                module_path = str(py_file.relative_to(self.root_path))
                
                # Determine module type
                module_type = self._determine_module_type(py_file)
                
                # Calculate basic metrics
                lines_of_code = self._count_lines_of_code(py_file)
                doc_score = self._calculate_documentation_score(py_file)
                complexity = self._calculate_module_complexity(py_file)
                
                self.modules[module_id] = ModuleNode(
                    id=module_id,
                    name=py_file.stem,
                    path=module_path,
                    lines_of_code=lines_of_code,
                    module_type=module_type,
                    documentation_score=doc_score,
                    complexity=complexity
                )
        
        print(f"📊 Discovered {len(self.modules)} modules")
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if file should be analyzed."""
        # Skip certain directories and files
        skip_patterns = {
            "__pycache__", ".git", ".pytest_cache", "node_modules",
            "venv", "env", ".venv", "build", "dist"
        }
        
        # Check if any parent directory should be skipped
        for part in file_path.parts:
            if part in skip_patterns:
                return False
        
        # Skip test files in this analysis (we'll handle them separately)
        if "test_" in file_path.name and file_path.name.startswith("test_"):
            return False
            
        return True
    
    def _get_module_id(self, file_path: Path) -> str:
        """Generate unique module ID from file path."""
        rel_path = file_path.relative_to(self.root_path)
        return str(rel_path).replace(os.sep, "/").replace(".py", "")
    
    def _determine_module_type(self, file_path: Path) -> str:
        """Determine the type of module based on path and content."""
        path_str = str(file_path)
        
        if "core/intelligence" in path_str:
            return "intelligence"
        elif "api" in path_str:
            return "api"
        elif "config" in path_str:
            return "config"
        elif "test" in path_str or file_path.name.startswith("test_"):
            return "test"
        elif "archive" in path_str:
            return "archive"
        else:
            return "standard"
    
    def _count_lines_of_code(self, file_path: Path) -> int:
        """Count lines of code excluding comments and empty lines."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            code_lines = 0
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    code_lines += 1
            
            return code_lines
        except Exception:
            return 0
    
    def _calculate_documentation_score(self, file_path: Path) -> float:
        """Calculate documentation score for module."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count docstrings and comments
            docstring_count = content.count('"""') // 2
            comment_count = content.count('#')
            total_lines = len(content.split('\n'))
            
            if total_lines == 0:
                return 0.0
            
            # Calculate score based on documentation density
            doc_score = min(1.0, (docstring_count * 5 + comment_count) / total_lines)
            return doc_score
        except Exception:
            return 0.0
    
    def _calculate_module_complexity(self, file_path: Path) -> float:
        """Calculate cyclomatic complexity of module."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                   ast.ExceptHandler, ast.With, ast.AsyncWith)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except Exception:
            return 1.0
    
    def _analyze_imports(self):
        """Analyze import relationships between modules."""
        print("🔗 Analyzing import relationships...")
        
        for module_id, module in self.modules.items():
            file_path = self.root_path / module.path
            imports = self._extract_imports(file_path)
            
            module.import_count = len(imports)
            
            for import_info in imports:
                target_module = self._resolve_import(import_info['module'], file_path)
                if target_module and target_module in self.modules:
                    # Add dependency
                    module.dependencies.add(target_module)
                    self.modules[target_module].dependents.add(module_id)
                    
                    # Create edge
                    edge = DependencyEdge(
                        source=module_id,
                        target=target_module,
                        import_type=import_info['type'],
                        coupling_strength=self._calculate_coupling_strength(import_info)
                    )
                    self.edges.append(edge)
    
    def _extract_imports(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract import statements from Python file."""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'module': alias.name,
                            'type': 'direct',
                            'line': node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append({
                            'module': node.module,
                            'type': 'from',
                            'line': node.lineno,
                            'names': [alias.name for alias in node.names]
                        })
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
        
        return imports
    
    def _resolve_import(self, import_name: str, current_file: Path) -> Optional[str]:
        """Resolve import to module ID."""
        # Handle relative imports
        if import_name.startswith('.'):
            current_dir = current_file.parent
            levels = len(import_name) - len(import_name.lstrip('.'))
            
            # Go up directory levels
            target_dir = current_dir
            for _ in range(levels - 1):
                target_dir = target_dir.parent
            
            # Build target path
            module_parts = import_name.lstrip('.').split('.')
            target_path = target_dir
            for part in module_parts:
                target_path = target_path / part
            
            # Check if it's a package or module
            if (target_path / "__init__.py").exists():
                target_path = target_path / "__init__.py"
            else:
                target_path = target_path.with_suffix(".py")
            
            if target_path.exists():
                return self._get_module_id(target_path)
        
        # Handle absolute imports within the project
        module_parts = import_name.split('.')
        
        # Try to find matching module
        for module_id in self.modules:
            module_path_parts = module_id.split('/')
            if module_path_parts[-1] == module_parts[-1]:
                return module_id
        
        return None
    
    def _calculate_coupling_strength(self, import_info: Dict[str, Any]) -> float:
        """Calculate coupling strength for import relationship."""
        base_strength = 1.0
        
        # Adjust based on import type
        if import_info['type'] == 'from':
            if 'names' in import_info:
                # More specific imports = stronger coupling
                base_strength += len(import_info['names']) * 0.1
        
        return min(5.0, base_strength)
    
    def _build_networkx_graph(self):
        """Build NetworkX graph for advanced analysis."""
        print("📊 Building NetworkX graph...")
        
        # Add nodes
        for module_id, module in self.modules.items():
            self.graph.add_node(module_id, **asdict(module))
        
        # Add edges
        for edge in self.edges:
            self.graph.add_edge(edge.source, edge.target, **asdict(edge))
    
    def _calculate_coupling_metrics(self):
        """Calculate coupling and cohesion metrics for modules."""
        print("📈 Calculating coupling metrics...")
        
        for module_id, module in self.modules.items():
            # Afferent coupling (Ca) - number of modules that depend on this module
            ca = len(module.dependents)
            
            # Efferent coupling (Ce) - number of modules this module depends on
            ce = len(module.dependencies)
            
            # Instability (I) = Ce / (Ca + Ce)
            instability = ce / (ca + ce) if (ca + ce) > 0 else 0
            
            # Update module with metrics
            module.coupling_score = instability
            
            # Calculate cohesion based on module type and complexity
            if module.module_type == "intelligence":
                # Intelligence modules should have high cohesion
                module.cohesion_score = min(1.0, 1.0 - (module.complexity - 1) / 50)
            else:
                module.cohesion_score = min(1.0, 1.0 - (module.complexity - 1) / 100)
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the module graph."""
        print("🔄 Detecting circular dependencies...")
        
        try:
            cycles = list(nx.simple_cycles(self.graph))
            
            # Mark edges involved in cycles
            for cycle in cycles:
                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]
                    
                    # Find and mark the edge
                    for edge in self.edges:
                        if edge.source == source and edge.target == target:
                            edge.dependency_type = "circular"
            
            return cycles
        except Exception as e:
            print(f"Warning: Could not detect cycles: {e}")
            return []
    
    def _identify_critical_modules(self) -> List[str]:
        """Identify critical modules based on centrality and importance."""
        print("⭐ Identifying critical modules...")
        
        critical_modules = []
        
        try:
            # Calculate centrality measures
            betweenness = nx.betweenness_centrality(self.graph)
            pagerank = nx.pagerank(self.graph)
            
            # Combine metrics to identify critical modules
            for module_id in self.modules:
                # Higher scores indicate more critical modules
                criticality_score = (
                    betweenness.get(module_id, 0) * 0.4 +
                    pagerank.get(module_id, 0) * 0.3 +
                    len(self.modules[module_id].dependents) * 0.3
                )
                
                # Mark intelligence modules as critical
                if any(intel_mod in module_id for intel_mod in self.intelligence_modules):
                    criticality_score += 0.5
                
                if criticality_score > 0.1:  # Threshold for criticality
                    critical_modules.append(module_id)
            
            # Sort by criticality (most critical first)
            critical_modules.sort(key=lambda x: (
                betweenness.get(x, 0) + pagerank.get(x, 0)
            ), reverse=True)
            
        except Exception as e:
            print(f"Warning: Could not calculate centrality: {e}")
            # Fallback to intelligence modules
            critical_modules = [m for m in self.modules if 
                             any(intel_mod in m for intel_mod in self.intelligence_modules)]
        
        return critical_modules[:20]  # Top 20 critical modules
    
    def _calculate_graph_metrics(self) -> Dict[str, Any]:
        """Calculate overall graph metrics."""
        print("📐 Calculating graph metrics...")
        
        try:
            metrics = {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'average_clustering': nx.average_clustering(self.graph.to_undirected()),
                'is_connected': nx.is_weakly_connected(self.graph),
                'number_of_components': nx.number_weakly_connected_components(self.graph)
            }
            
            # Calculate centralization
            if self.graph.number_of_nodes() > 1:
                degree_centrality = nx.degree_centrality(self.graph)
                max_centrality = max(degree_centrality.values())
                metrics['degree_centralization'] = max_centrality
            else:
                metrics['degree_centralization'] = 0
            
            return metrics
        except Exception as e:
            print(f"Warning: Could not calculate graph metrics: {e}")
            return {
                'total_nodes': len(self.modules),
                'total_edges': len(self.edges),
                'density': 0,
                'average_clustering': 0,
                'is_connected': False,
                'number_of_components': len(self.modules)
            }
    
    def _generate_insights(self) -> List[str]:
        """Generate insights from dependency analysis."""
        insights = []
        
        # Coupling insights
        high_coupling_modules = [
            m for m in self.modules.values() 
            if m.coupling_score > 0.7
        ]
        if high_coupling_modules:
            insights.append(f"Found {len(high_coupling_modules)} modules with high coupling (>0.7)")
        
        # Complexity insights
        complex_modules = [
            m for m in self.modules.values()
            if m.complexity > 20
        ]
        if complex_modules:
            insights.append(f"Found {len(complex_modules)} highly complex modules (complexity > 20)")
        
        # Documentation insights
        poorly_documented = [
            m for m in self.modules.values()
            if m.documentation_score < 0.3
        ]
        if poorly_documented:
            insights.append(f"Found {len(poorly_documented)} poorly documented modules (score < 0.3)")
        
        # Intelligence module insights
        intelligence_modules = [
            m for m in self.modules.values()
            if m.module_type == "intelligence"
        ]
        insights.append(f"Intelligence framework contains {len(intelligence_modules)} specialized modules")
        
        return insights
    
    def _prepare_neo4j_export(self) -> Dict[str, Any]:
        """Prepare data for Neo4j export."""
        neo4j_data = {
            'nodes': [],
            'relationships': [],
            'constraints': [],
            'indexes': []
        }
        
        # Create nodes
        for module_id, module in self.modules.items():
            node = {
                'labels': ['Module', module.module_type.title()],
                'properties': {
                    'id': module_id,
                    'name': module.name,
                    'path': module.path,
                    'lines_of_code': module.lines_of_code,
                    'complexity': module.complexity,
                    'coupling_score': module.coupling_score,
                    'cohesion_score': module.cohesion_score,
                    'documentation_score': module.documentation_score,
                    'import_count': module.import_count,
                    'module_type': module.module_type
                }
            }
            neo4j_data['nodes'].append(node)
        
        # Create relationships
        for edge in self.edges:
            relationship = {
                'type': 'DEPENDS_ON',
                'source': edge.source,
                'target': edge.target,
                'properties': {
                    'import_type': edge.import_type,
                    'coupling_strength': edge.coupling_strength,
                    'dependency_type': edge.dependency_type
                }
            }
            neo4j_data['relationships'].append(relationship)
        
        # Add constraints and indexes for performance
        neo4j_data['constraints'] = [
            'CREATE CONSTRAINT module_id_unique IF NOT EXISTS FOR (m:Module) REQUIRE m.id IS UNIQUE'
        ]
        
        neo4j_data['indexes'] = [
            'CREATE INDEX module_type_index IF NOT EXISTS FOR (m:Module) ON (m.module_type)',
            'CREATE INDEX coupling_score_index IF NOT EXISTS FOR (m:Module) ON (m.coupling_score)'
        ]
        
        return neo4j_data
    
    def export_results(self, output_file: str = "dependency_analysis_results.json"):
        """Export analysis results to JSON file."""
        output_path = self.root_path / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"📄 Results exported to {output_path}")
        return output_path


def main():
    """Main analysis execution."""
    print("🚀 TestMaster Dependency Graph Analysis - Agent B Phase 2")
    print("=" * 60)
    
    analyzer = DependencyGraphAnalyzer()
    results = analyzer.analyze_module_dependencies()
    
    # Export results
    analyzer.export_results()
    
    # Print summary
    print("\n📊 ANALYSIS SUMMARY:")
    print(f"Total modules analyzed: {results['metadata']['total_modules']}")
    print(f"Total dependencies: {results['metadata']['total_dependencies']}")
    print(f"Circular dependencies: {len(results['circular_dependencies'])}")
    print(f"Critical modules identified: {len(results['critical_modules'])}")
    print(f"Graph density: {results['graph_metrics'].get('density', 0):.4f}")
    
    print("\n🎯 KEY INSIGHTS:")
    for insight in results['insights']:
        print(f"  • {insight}")
    
    print("\n✅ Dependency analysis complete!")
    return results


if __name__ == "__main__":
    main()