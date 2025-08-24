#!/usr/bin/env python3
"""
Functional Linkage Analyzer for Codebase Surveillance
=====================================================

Analyzes functional relationships between Python files to identify:
- Hanging/orphaned files (no incoming or outgoing dependencies)
- Marginal files (weakly connected to the codebase)
- Strongly connected components and clusters
- File coupling strength and centrality measures

Provides exhaustive mapping for Neo4j visualization and dashboard display.

Author: Claude Code - Based on existing TestMaster analysis capabilities
"""

import ast
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util

@dataclass
class FileNode:
    """Represents a Python file in the dependency graph."""
    path: str
    name: str
    size_lines: int
    imports: List[str]
    exports: List[str]
    functions: List[str]
    classes: List[str]
    incoming_deps: int
    outgoing_deps: int
    coupling_score: float
    centrality_score: float
    cluster_id: Optional[str]
    connectivity_status: str  # "orphaned", "marginal", "weakly_connected", "strongly_connected"

@dataclass
class DependencyEdge:
    """Represents a dependency relationship between files."""
    source_file: str
    target_file: str
    relationship_type: str  # "import", "from_import", "function_call", "class_inheritance"
    strength: float
    line_number: int

class FunctionalLinkageAnalyzer:
    """Analyzes functional linkages and identifies hanging/marginal files."""
    
    def __init__(self, base_directory: str = ".", exclude_patterns: List[str] = None):
        self.base_directory = Path(base_directory)
        self.exclude_patterns = exclude_patterns or [
            "__pycache__", ".git", ".pytest_cache", "node_modules",
            "venv", "env", ".venv", "dist", "build", "*.egg-info"
        ]
        
        # Analysis results
        self.file_nodes: Dict[str, FileNode] = {}
        self.dependency_edges: List[DependencyEdge] = []
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Connectivity analysis
        self.orphaned_files: Set[str] = set()
        self.marginal_files: Set[str] = set()
        self.hanging_files: Set[str] = set()
        self.strongly_connected_clusters: List[Set[str]] = []
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive functional linkage analysis."""
        print("Starting Functional Linkage Analysis...")
        
        # Step 1: Discover all Python files
        python_files = self._discover_python_files()
        print(f"Found {len(python_files)} Python files")
        
        # Step 2: Analyze each file individually
        print("Analyzing individual files...")
        for py_file in python_files:
            self._analyze_file(py_file)
        
        # Step 3: Build dependency graph
        print("Building dependency graph...")
        self._build_dependency_graph()
        
        # Step 4: Calculate connectivity metrics
        print("Calculating connectivity metrics...")
        self._calculate_connectivity_metrics()
        
        # Step 5: Identify hanging/marginal files
        print("Identifying hanging and marginal files...")
        self._identify_connectivity_issues()
        
        # Step 6: Find strongly connected components
        print("Finding strongly connected components...")
        self._find_connected_components()
        
        # Step 7: Generate comprehensive report
        print("Generating comprehensive linkage report...")
        return self._generate_comprehensive_report()
    
    def _discover_python_files(self) -> List[Path]:
        """Discover all Python files in the codebase."""
        python_files = []
        
        for root, dirs, files in os.walk(self.base_directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    # Skip excluded patterns
                    if not any(pattern in str(file_path) for pattern in self.exclude_patterns):
                        python_files.append(file_path)
        
        return python_files
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for imports, exports, and structure."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.count('\\n') + 1
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract file information
            relative_path = str(file_path.relative_to(self.base_directory))
            file_name = file_path.stem
            
            imports = self._extract_imports(tree)
            exports = self._extract_exports(tree)
            functions = self._extract_functions(tree)
            classes = self._extract_classes(tree)
            
            # Create file node
            self.file_nodes[relative_path] = FileNode(
                path=relative_path,
                name=file_name,
                size_lines=lines,
                imports=imports,
                exports=exports,
                functions=functions,
                classes=classes,
                incoming_deps=0,  # Will be calculated later
                outgoing_deps=len(imports),
                coupling_score=0.0,  # Will be calculated later
                centrality_score=0.0,  # Will be calculated later
                cluster_id=None,
                connectivity_status="unknown"
            )
            
        except Exception as e:
            print(f"Warning: Error analyzing {file_path}: {e}")
    
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
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        return imports
    
    def _extract_exports(self, tree: ast.AST) -> List[str]:
        """Extract exported functions and classes."""
        exports = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith('_'):  # Public exports
                    exports.append(node.name)
        
        return exports
    
    def _extract_functions(self, tree: ast.AST) -> List[str]:
        """Extract function definitions."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
        
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[str]:
        """Extract class definitions."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return classes
    
    def _build_dependency_graph(self) -> None:
        """Build dependency graph from file imports."""
        for source_path, file_node in self.file_nodes.items():
            for import_name in file_node.imports:
                # Try to resolve import to a file in our codebase
                target_path = self._resolve_import_to_file(import_name, source_path)
                
                if target_path and target_path in self.file_nodes:
                    # Create dependency edge
                    edge = DependencyEdge(
                        source_file=source_path,
                        target_file=target_path,
                        relationship_type="import",
                        strength=1.0,
                        line_number=0  # Would need more detailed AST analysis
                    )
                    
                    self.dependency_edges.append(edge)
                    self.dependency_graph[source_path].add(target_path)
                    self.reverse_graph[target_path].add(source_path)
    
    def _resolve_import_to_file(self, import_name: str, source_path: str) -> Optional[str]:
        """Resolve an import name to a file path in the codebase."""
        # Try different resolution strategies
        
        # Strategy 1: Direct module name to file
        potential_paths = [
            f"{import_name.replace('.', os.sep)}.py",
            f"{import_name.replace('.', os.sep)}/__init__.py",
        ]
        
        # Strategy 2: Relative to source file directory
        source_dir = Path(source_path).parent
        for potential in potential_paths:
            full_path = source_dir / potential
            relative_path = str(full_path)
            if relative_path in self.file_nodes:
                return relative_path
        
        # Strategy 3: Absolute from base directory
        for potential in potential_paths:
            if potential in self.file_nodes:
                return potential
        
        # Strategy 4: Search for files with matching names
        import_parts = import_name.split('.')
        if import_parts:
            search_name = import_parts[-1]
            for file_path in self.file_nodes:
                if Path(file_path).stem == search_name:
                    return file_path
        
        return None
    
    def _calculate_connectivity_metrics(self) -> None:
        """Calculate connectivity metrics for each file."""
        for file_path, file_node in self.file_nodes.items():
            # Update dependency counts
            file_node.incoming_deps = len(self.reverse_graph[file_path])
            file_node.outgoing_deps = len(self.dependency_graph[file_path])
            
            # Calculate coupling score (normalized)
            total_deps = file_node.incoming_deps + file_node.outgoing_deps
            max_possible = len(self.file_nodes) - 1  # Can connect to all other files
            file_node.coupling_score = total_deps / max_possible if max_possible > 0 else 0.0
            
            # Calculate centrality score (simple betweenness approximation)
            file_node.centrality_score = self._calculate_centrality(file_path)
    
    def _calculate_centrality(self, file_path: str) -> float:
        """Calculate centrality score for a file (simplified PageRank-like algorithm)."""
        # Simple centrality based on in-degree and out-degree
        incoming = len(self.reverse_graph[file_path])
        outgoing = len(self.dependency_graph[file_path])
        
        # Weight incoming connections more (files that are imported are more central)
        centrality = (incoming * 2 + outgoing) / (len(self.file_nodes) + 1)
        return min(centrality, 1.0)
    
    def _identify_connectivity_issues(self) -> None:
        """Identify orphaned, hanging, and marginal files."""
        for file_path, file_node in self.file_nodes.items():
            incoming = file_node.incoming_deps
            outgoing = file_node.outgoing_deps
            
            # Orphaned files: No incoming or outgoing dependencies
            if incoming == 0 and outgoing == 0:
                self.orphaned_files.add(file_path)
                file_node.connectivity_status = "orphaned"
            
            # Hanging files: No incoming dependencies (nothing imports them)
            elif incoming == 0 and outgoing > 0:
                self.hanging_files.add(file_path)
                file_node.connectivity_status = "hanging"
            
            # Marginal files: Very few connections
            elif incoming + outgoing <= 2:
                self.marginal_files.add(file_path)
                file_node.connectivity_status = "marginal"
            
            # Weakly connected: Few connections relative to codebase size
            elif file_node.coupling_score < 0.1:
                file_node.connectivity_status = "weakly_connected"
            
            # Strongly connected: Many connections
            else:
                file_node.connectivity_status = "strongly_connected"
    
    def _find_connected_components(self) -> None:
        """Find strongly connected components using Tarjan's algorithm."""
        # Simplified connected components using DFS
        visited = set()
        components = []
        
        def dfs(node: str, component: Set[str]) -> None:
            if node in visited:
                return
            
            visited.add(node)
            component.add(node)
            
            # Follow outgoing edges
            for neighbor in self.dependency_graph[node]:
                dfs(neighbor, component)
            
            # Follow incoming edges
            for neighbor in self.reverse_graph[node]:
                dfs(neighbor, component)
        
        for file_path in self.file_nodes:
            if file_path not in visited:
                component = set()
                dfs(file_path, component)
                if len(component) > 1:  # Only store multi-file components
                    components.append(component)
                    
                    # Assign cluster IDs
                    cluster_id = f"cluster_{len(components)}"
                    for file in component:
                        if file in self.file_nodes:
                            self.file_nodes[file].cluster_id = cluster_id
        
        self.strongly_connected_clusters = components
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive linkage analysis report."""
        total_files = len(self.file_nodes)
        
        # Summary statistics
        summary = {
            "total_files": total_files,
            "total_dependencies": len(self.dependency_edges),
            "orphaned_files_count": len(self.orphaned_files),
            "hanging_files_count": len(self.hanging_files),
            "marginal_files_count": len(self.marginal_files),
            "connected_components": len(self.strongly_connected_clusters),
            "average_coupling": sum(node.coupling_score for node in self.file_nodes.values()) / total_files if total_files > 0 else 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Detailed results
        detailed_results = {
            "summary": summary,
            "file_nodes": {path: asdict(node) for path, node in self.file_nodes.items()},
            "dependency_edges": [asdict(edge) for edge in self.dependency_edges],
            "connectivity_issues": {
                "orphaned_files": list(self.orphaned_files),
                "hanging_files": list(self.hanging_files),
                "marginal_files": list(self.marginal_files)
            },
            "connected_components": [list(component) for component in self.strongly_connected_clusters],
            "centrality_ranking": self._get_centrality_ranking(),
            "coupling_ranking": self._get_coupling_ranking()
        }
        
        return detailed_results
    
    def _get_centrality_ranking(self) -> List[Dict[str, Any]]:
        """Get files ranked by centrality score."""
        ranked = sorted(
            self.file_nodes.items(),
            key=lambda x: x[1].centrality_score,
            reverse=True
        )
        
        return [
            {
                "file_path": path,
                "centrality_score": node.centrality_score,
                "incoming_deps": node.incoming_deps,
                "outgoing_deps": node.outgoing_deps
            }
            for path, node in ranked[:20]  # Top 20
        ]
    
    def _get_coupling_ranking(self) -> List[Dict[str, Any]]:
        """Get files ranked by coupling score."""
        ranked = sorted(
            self.file_nodes.items(),
            key=lambda x: x[1].coupling_score,
            reverse=True
        )
        
        return [
            {
                "file_path": path,
                "coupling_score": node.coupling_score,
                "total_connections": node.incoming_deps + node.outgoing_deps,
                "connectivity_status": node.connectivity_status
            }
            for path, node in ranked[:20]  # Top 20
        ]
    
    def generate_neo4j_compatible_graph(self) -> Dict[str, Any]:
        """Generate Neo4j compatible graph data."""
        nodes = []
        relationships = []
        
        # Create nodes for files
        for file_path, file_node in self.file_nodes.items():
            nodes.append({
                "id": file_path,
                "labels": ["File", "PythonFile"],
                "properties": {
                    "name": file_node.name,
                    "path": file_node.path,
                    "size_lines": file_node.size_lines,
                    "connectivity_status": file_node.connectivity_status,
                    "coupling_score": file_node.coupling_score,
                    "centrality_score": file_node.centrality_score,
                    "incoming_deps": file_node.incoming_deps,
                    "outgoing_deps": file_node.outgoing_deps,
                    "cluster_id": file_node.cluster_id,
                    "functions_count": len(file_node.functions),
                    "classes_count": len(file_node.classes)
                }
            })
        
        # Create relationships
        for edge in self.dependency_edges:
            relationships.append({
                "source_id": edge.source_file,
                "target_id": edge.target_file,
                "type": "DEPENDS_ON",
                "properties": {
                    "relationship_type": edge.relationship_type,
                    "strength": edge.strength,
                    "line_number": edge.line_number
                }
            })
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "analysis_type": "functional_linkage",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def save_results(self, output_dir: str = ".") -> None:
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save comprehensive report
        report = self._generate_comprehensive_report()
        with open(output_path / "functional_linkage_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save Neo4j compatible graph
        neo4j_graph = self.generate_neo4j_compatible_graph()
        with open(output_path / "functional_linkage_graph.json", 'w') as f:
            json.dump(neo4j_graph, f, indent=2)
        
        # Save human-readable summary
        self._save_readable_summary(output_path / "linkage_summary.txt", report)
        
        print(f"Results saved to {output_path}")
    
    def _save_readable_summary(self, file_path: Path, report: Dict[str, Any]) -> None:
        """Save a human-readable summary."""
        with open(file_path, 'w') as f:
            f.write("FUNCTIONAL LINKAGE ANALYSIS SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            
            summary = report["summary"]
            f.write(f"Total Files Analyzed: {summary['total_files']}\\n")
            f.write(f"Total Dependencies: {summary['total_dependencies']}\\n")
            f.write(f"Average Coupling Score: {summary['average_coupling']:.3f}\\n\\n")
            
            f.write("CONNECTIVITY ISSUES:\\n")
            f.write(f"  Orphaned Files: {summary['orphaned_files_count']}\\n")
            f.write(f"  Hanging Files: {summary['hanging_files_count']}\\n")
            f.write(f"  Marginal Files: {summary['marginal_files_count']}\\n\\n")
            
            # List orphaned files
            if report["connectivity_issues"]["orphaned_files"]:
                f.write("ORPHANED FILES (no dependencies in/out):\\n")
                for file_path in report["connectivity_issues"]["orphaned_files"]:
                    f.write(f"  - {file_path}\\n")
                f.write("\\n")
            
            # List hanging files
            if report["connectivity_issues"]["hanging_files"]:
                f.write("HANGING FILES (nothing imports them):\\n")
                for file_path in report["connectivity_issues"]["hanging_files"]:
                    f.write(f"  - {file_path}\\n")
                f.write("\\n")
            
            # List marginal files
            if report["connectivity_issues"]["marginal_files"]:
                f.write("MARGINAL FILES (weakly connected):\\n")
                for file_path in report["connectivity_issues"]["marginal_files"]:
                    f.write(f"  - {file_path}\\n")
                f.write("\\n")
            
            # Top central files
            f.write("TOP CENTRAL FILES:\\n")
            for rank in report["centrality_ranking"][:10]:
                f.write(f"  - {rank['file_path']} (score: {rank['centrality_score']:.3f})\\n")

def main():
    """Run functional linkage analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze functional linkages in Python codebase")
    parser.add_argument("--directory", "-d", default=".", help="Directory to analyze")
    parser.add_argument("--output", "-o", default=".", help="Output directory for results")
    args = parser.parse_args()
    
    analyzer = FunctionalLinkageAnalyzer(args.directory)
    analyzer.analyze_codebase()
    analyzer.save_results(args.output)
    
    print("Functional linkage analysis complete!")

if __name__ == "__main__":
    main()