#!/usr/bin/env python3
"""
Dependency Mapping Script
========================

Maps all dependencies and integration points between components.
This script ONLY analyzes - it NEVER removes anything.

Part of Phase 1: Comprehensive Analysis & Mapping
"""

import ast
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt


@dataclass 
class DependencyEdge:
    """Represents a dependency relationship between components."""
    source: str
    target: str
    dependency_type: str  # import, inheritance, function_call, etc.
    line_number: int
    context: str  # The actual code that creates the dependency


@dataclass
class ComponentDependencies:
    """Complete dependency information for a component."""
    component_path: str
    imports: List[DependencyEdge]
    calls: List[DependencyEdge] 
    inheritance: List[DependencyEdge]
    composition: List[DependencyEdge]
    
    # Dependency statistics
    internal_deps: Set[str] = field(default_factory=set)
    external_deps: Set[str] = field(default_factory=set)
    circular_deps: List[str] = field(default_factory=list)
    
    # Integration points
    api_endpoints: List[str] = field(default_factory=list)
    event_handlers: List[str] = field(default_factory=list)
    config_dependencies: List[str] = field(default_factory=list)


class DependencyAnalyzer:
    """
    Analyzes dependencies and integration points between components.
    Critical for understanding how to safely consolidate without breaking anything.
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.dependencies: Dict[str, ComponentDependencies] = {}
        self.dependency_graph = nx.DiGraph()
        
        # Load component analysis results
        analysis_file = base_path / "phase1_component_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                self.component_analysis = json.load(f)
        else:
            self.component_analysis = {}
        
        print(f"[INFO] Dependency Analyzer initialized for: {base_path}")
    
    def analyze_dependencies(self, components: List[str] = None) -> Dict[str, ComponentDependencies]:
        """Analyze dependencies for specified components or all components."""
        print(f"[INFO] Starting dependency analysis...")
        start_time = time.time()
        
        if components is None:
            # Use components from previous analysis
            if 'components' in self.component_analysis:
                components = list(self.component_analysis['components'].keys())
            else:
                print("[ERROR] No component analysis found. Run analyze_components.py first.")
                return {}
        
        for component_path in components:
            try:
                full_path = self.base_path / component_path
                if full_path.exists():
                    deps = self._analyze_component_dependencies(full_path, component_path)
                    if deps:
                        self.dependencies[component_path] = deps
                        self._add_to_dependency_graph(component_path, deps)
            except Exception as e:
                print(f"[ERROR] Failed to analyze dependencies for {component_path}: {e}")
        
        # Detect circular dependencies
        self._detect_circular_dependencies()
        
        duration = time.time() - start_time
        print(f"[INFO] Dependency analysis complete: {len(self.dependencies)} components analyzed in {duration:.2f}s")
        
        return self.dependencies
    
    def _analyze_component_dependencies(self, file_path: Path, component_path: str) -> Optional[ComponentDependencies]:
        """Analyze dependencies for a single component."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            deps = ComponentDependencies(
                component_path=component_path,
                imports=[],
                calls=[],
                inheritance=[],
                composition=[]
            )
            
            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    dep_edge = self._analyze_import_dependency(node, component_path, content)
                    if dep_edge:
                        deps.imports.append(dep_edge)
                
                elif isinstance(node, ast.ClassDef):
                    # Analyze inheritance
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            deps.inheritance.append(DependencyEdge(
                                source=component_path,
                                target=base.id,
                                dependency_type="inheritance",
                                line_number=node.lineno,
                                context=f"class {node.name}({base.id})"
                            ))
                
                elif isinstance(node, ast.Call):
                    # Analyze function calls that might be cross-component
                    call_info = self._analyze_function_call(node, component_path)
                    if call_info:
                        deps.calls.append(call_info)
            
            # Classify dependencies
            self._classify_dependencies(deps)
            
            # Detect integration points
            self._detect_integration_points(content, deps)
            
            return deps
            
        except Exception as e:
            print(f"[ERROR] Error analyzing dependencies for {file_path}: {e}")
            return None
    
    def _analyze_import_dependency(self, node, source_component: str, content: str) -> Optional[DependencyEdge]:
        """Analyze an import statement to create dependency edge."""
        try:
            context_lines = content.splitlines()
            context = context_lines[node.lineno - 1] if node.lineno <= len(context_lines) else ""
            
            if isinstance(node, ast.Import):
                target = node.names[0].name
            else:  # ImportFrom
                target = node.module or ""
            
            return DependencyEdge(
                source=source_component,
                target=target,
                dependency_type="import",
                line_number=node.lineno,
                context=context.strip()
            )
        except Exception:
            return None
    
    def _analyze_function_call(self, node: ast.Call, source_component: str) -> Optional[DependencyEdge]:
        """Analyze function calls for cross-component dependencies."""
        try:
            # Look for calls that might be to other components
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    target = f"{node.func.value.id}.{node.func.attr}"
                    return DependencyEdge(
                        source=source_component,
                        target=target,
                        dependency_type="function_call",
                        line_number=node.lineno,
                        context=f"call to {target}"
                    )
        except Exception:
            pass
        return None
    
    def _classify_dependencies(self, deps: ComponentDependencies):
        """Classify dependencies as internal or external."""
        testmaster_keywords = ['testmaster', 'core', 'integration', 'dashboard']
        
        for import_dep in deps.imports:
            target = import_dep.target.lower()
            if any(keyword in target for keyword in testmaster_keywords) or target.startswith('.'):
                deps.internal_deps.add(import_dep.target)
            else:
                deps.external_deps.add(import_dep.target)
    
    def _detect_integration_points(self, content: str, deps: ComponentDependencies):
        """Detect integration points like API endpoints, event handlers, etc."""
        content_lower = content.lower()
        
        # API endpoints
        api_patterns = ['@app.route', '@blueprint.route', 'flask', 'fastapi', 'endpoint']
        for pattern in api_patterns:
            if pattern in content_lower:
                deps.api_endpoints.append(pattern)
        
        # Event handlers
        event_patterns = ['@event', 'on_', 'handle_', 'listener', 'observer']
        for pattern in event_patterns:
            if pattern in content_lower:
                deps.event_handlers.append(pattern)
        
        # Config dependencies
        config_patterns = ['config', 'settings', 'env', 'environment']
        for pattern in config_patterns:
            if pattern in content_lower:
                deps.config_dependencies.append(pattern)
    
    def _add_to_dependency_graph(self, component: str, deps: ComponentDependencies):
        """Add component dependencies to the graph."""
        self.dependency_graph.add_node(component)
        
        for dep_edge in deps.imports + deps.calls + deps.inheritance:
            if any(keyword in dep_edge.target.lower() for keyword in ['testmaster', 'core', 'integration', 'dashboard']):
                self.dependency_graph.add_edge(component, dep_edge.target, 
                                             type=dep_edge.dependency_type,
                                             line=dep_edge.line_number)
    
    def _detect_circular_dependencies(self):
        """Detect circular dependencies in the component graph."""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            for component_path, deps in self.dependencies.items():
                for cycle in cycles:
                    if component_path in cycle:
                        deps.circular_deps.extend([c for c in cycle if c != component_path])
        except Exception as e:
            print(f"[WARN] Error detecting circular dependencies: {e}")
    
    def generate_consolidation_impact_analysis(self) -> Dict[str, Any]:
        """Generate analysis of consolidation impact based on dependencies."""
        impact_analysis = {
            'high_risk_components': [],
            'safe_to_consolidate': [],
            'integration_challenges': [],
            'recommended_consolidation_order': []
        }
        
        # Analyze each component's consolidation risk
        for component_path, deps in self.dependencies.items():
            risk_factors = []
            
            # High external dependencies = higher risk
            if len(deps.external_deps) > 10:
                risk_factors.append("many_external_deps")
            
            # Circular dependencies = higher risk
            if deps.circular_deps:
                risk_factors.append("circular_deps")
            
            # API endpoints = integration challenges
            if deps.api_endpoints:
                risk_factors.append("has_api_endpoints")
            
            # Complex internal dependencies = higher risk
            if len(deps.internal_deps) > 15:
                risk_factors.append("complex_internal_deps")
            
            component_info = {
                'path': component_path,
                'risk_factors': risk_factors,
                'internal_deps': len(deps.internal_deps),
                'external_deps': len(deps.external_deps),
                'has_api_endpoints': bool(deps.api_endpoints),
                'has_circular_deps': bool(deps.circular_deps)
            }
            
            if len(risk_factors) >= 3:
                impact_analysis['high_risk_components'].append(component_info)
            elif len(risk_factors) <= 1:
                impact_analysis['safe_to_consolidate'].append(component_info)
            
            if deps.api_endpoints or deps.event_handlers:
                impact_analysis['integration_challenges'].append({
                    'component': component_path,
                    'api_endpoints': deps.api_endpoints,
                    'event_handlers': deps.event_handlers
                })
        
        # Recommend consolidation order (safest first)
        all_components = [(path, len(deps.internal_deps) + len(deps.external_deps)) 
                         for path, deps in self.dependencies.items()]
        all_components.sort(key=lambda x: x[1])  # Sort by dependency count
        
        impact_analysis['recommended_consolidation_order'] = [comp[0] for comp in all_components]
        
        return impact_analysis
    
    def visualize_dependencies(self, output_file: str = "dependency_graph.png"):
        """Generate a visual representation of the dependency graph."""
        try:
            plt.figure(figsize=(16, 12))
            
            # Use different layouts for different graph sizes
            if len(self.dependency_graph.nodes) < 20:
                pos = nx.spring_layout(self.dependency_graph, k=3, iterations=50)
            else:
                pos = nx.circular_layout(self.dependency_graph)
            
            # Color nodes by component type
            node_colors = []
            for node in self.dependency_graph.nodes():
                if 'analytics' in node.lower():
                    node_colors.append('lightblue')
                elif 'test' in node.lower():
                    node_colors.append('lightgreen') 
                elif 'integration' in node.lower():
                    node_colors.append('lightyellow')
                else:
                    node_colors.append('lightgray')
            
            # Draw the graph
            nx.draw(self.dependency_graph, pos, 
                   node_color=node_colors,
                   node_size=500,
                   font_size=8,
                   font_weight='bold',
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   alpha=0.7)
            
            plt.title("Component Dependency Graph", fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = self.base_path / output_file
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Dependency graph saved to: {output_path}")
            
        except Exception as e:
            print(f"[WARN] Could not generate dependency visualization: {e}")
    
    def generate_report(self, output_file: str = "dependency_analysis_report.json"):
        """Generate comprehensive dependency analysis report."""
        
        # Generate consolidation impact analysis
        impact_analysis = self.generate_consolidation_impact_analysis()
        
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_components': len(self.dependencies),
                'total_dependencies': sum(len(deps.imports) + len(deps.calls) + len(deps.inheritance) 
                                        for deps in self.dependencies.values())
            },
            'dependency_statistics': {
                'components_with_circular_deps': len([d for d in self.dependencies.values() if d.circular_deps]),
                'components_with_api_endpoints': len([d for d in self.dependencies.values() if d.api_endpoints]),
                'average_internal_deps': sum(len(d.internal_deps) for d in self.dependencies.values()) / len(self.dependencies) if self.dependencies else 0,
                'average_external_deps': sum(len(d.external_deps) for d in self.dependencies.values()) / len(self.dependencies) if self.dependencies else 0
            },
            'consolidation_impact_analysis': impact_analysis,
            'detailed_dependencies': {path: asdict(deps) for path, deps in self.dependencies.items()}
        }
        
        # Write report
        output_path = self.base_path / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[INFO] Dependency analysis report generated: {output_path}")
        return report


def main():
    """Main execution function."""
    print("=" * 80)
    print("DEPENDENCY ANALYSIS - PHASE 1: INTEGRATION POINT MAPPING")
    print("=" * 80)
    
    # Initialize analyzer
    base_path = Path(".")
    analyzer = DependencyAnalyzer(base_path)
    
    # Run dependency analysis
    print(f"[INFO] Analyzing dependencies for all components...")
    dependencies = analyzer.analyze_dependencies()
    
    # Generate visualization
    analyzer.visualize_dependencies("phase1_dependency_graph.png")
    
    # Generate comprehensive report
    report = analyzer.generate_report("phase1_dependency_analysis.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("DEPENDENCY ANALYSIS SUMMARY")
    print("=" * 80)
    
    metadata = report['analysis_metadata']
    stats = report['dependency_statistics']
    impact = report['consolidation_impact_analysis']
    
    print(f"Components analyzed: {metadata['total_components']}")
    print(f"Total dependencies: {metadata['total_dependencies']}")
    print(f"Components with circular deps: {stats['components_with_circular_deps']}")
    print(f"Components with API endpoints: {stats['components_with_api_endpoints']}")
    print(f"Average internal deps per component: {stats['average_internal_deps']:.1f}")
    print(f"Average external deps per component: {stats['average_external_deps']:.1f}")
    
    print(f"\nCONSOLIDATION IMPACT ANALYSIS:")
    print(f"High-risk components: {len(impact['high_risk_components'])}")
    print(f"Safe to consolidate: {len(impact['safe_to_consolidate'])}")
    print(f"Integration challenges: {len(impact['integration_challenges'])}")
    
    print(f"\n[INFO] Detailed analysis saved to: phase1_dependency_analysis.json")
    print("[INFO] Dependency graph saved to: phase1_dependency_graph.png")
    print("[INFO] Phase 1 dependency mapping complete!")
    
    return dependencies


if __name__ == '__main__':
    main()