"""
Hierarchical Refactoring Analyzer for Massive Codebases

Handles 100k-200k+ LOC codebases through:
- Multi-level hierarchical analysis
- Progressive summarization
- Functional clustering
- Chunk-based refactoring roadmaps
- Dependency tracking
"""

import ast
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

@dataclass
class CodeMetrics:
    """Metrics for a code unit (file, class, function)."""
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    dependencies: Set[str] = field(default_factory=set)
    coupling_score: float = 0.0
    cohesion_score: float = 0.0
    test_coverage: float = 0.0
    last_modified: datetime = field(default_factory=datetime.now)
    
@dataclass
class RefactorOpportunity:
    """A specific refactoring opportunity."""
    id: str
    type: str  # 'extract_method', 'split_class', 'remove_duplication', etc.
    severity: str  # 'high', 'medium', 'low'
    location: str  # file:line
    description: str
    estimated_effort: int  # hours
    impact_score: float  # 0-100
    dependencies: List[str] = field(default_factory=list)
    
@dataclass
class FunctionalCluster:
    """A cluster of related code units."""
    id: str
    name: str
    files: List[str]
    primary_purpose: str
    metrics: CodeMetrics
    refactor_opportunities: List[RefactorOpportunity] = field(default_factory=list)
    sub_clusters: List['FunctionalCluster'] = field(default_factory=list)
    
@dataclass
class CodebaseHierarchy:
    """Hierarchical representation of the codebase."""
    root_path: str
    total_lines: int
    total_files: int
    clusters: List[FunctionalCluster]
    global_metrics: CodeMetrics
    summary: str
    
@dataclass
class RefactorRoadmap:
    """A roadmap for refactoring efforts."""
    id: str
    title: str
    phases: List[Dict[str, Any]]
    total_effort_hours: int
    priority_score: float
    dependencies_graph: Dict[str, List[str]]
    risk_assessment: str

class HierarchicalRefactoringAnalyzer:
    """
    Main analyzer for hierarchical refactoring of massive codebases.
    
    Features:
    - Lightweight first-pass scanning
    - Functional clustering
    - Progressive summarization
    - Chunk-based analysis
    - Dependency tracking
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Caches for performance
        self.file_cache: Dict[str, Dict[str, Any]] = {}
        self.cluster_cache: Dict[str, FunctionalCluster] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Analysis state
        self.current_hierarchy: Optional[CodebaseHierarchy] = None
        self.refactor_opportunities: List[RefactorOpportunity] = []
        self.analysis_lock = threading.Lock()
        
        # Configuration
        self.chunk_size = 10000  # Lines per chunk
        self.max_file_size = 50000  # Max lines to analyze in detail
        self.clustering_threshold = 0.7  # Similarity threshold for clustering
        
    def analyze_codebase(self, root_path: str, codebase_name: str = None) -> CodebaseHierarchy:
        """
        Analyze entire codebase hierarchically.
        
        This is the main entry point for analyzing massive codebases.
        """
        print(f"Starting hierarchical analysis of {root_path}")
        
        # Phase 1: Lightweight metadata scan
        metadata = self._scan_metadata(root_path)
        print(f"Phase 1 complete: {metadata['total_files']} files, {metadata['total_lines']} lines")
        
        # Phase 2: Functional clustering
        clusters = self._create_functional_clusters(metadata)
        print(f"Phase 2 complete: {len(clusters)} top-level clusters identified")
        
        # Phase 3: Progressive analysis of clusters
        analyzed_clusters = self._analyze_clusters_progressively(clusters)
        print(f"Phase 3 complete: Detailed analysis of clusters")
        
        # Phase 4: Generate hierarchy
        hierarchy = self._build_hierarchy(root_path, metadata, analyzed_clusters)
        self.current_hierarchy = hierarchy
        print(f"Phase 4 complete: Hierarchy built")
        
        return hierarchy
    
    def _scan_metadata(self, root_path: str) -> Dict[str, Any]:
        """
        Phase 1: Lightweight metadata scanning.
        
        Quickly scan all files to gather basic metadata without deep parsing.
        """
        metadata = {
            'files': [],
            'total_lines': 0,
            'total_files': 0,
            'file_types': defaultdict(int),
            'module_structure': defaultdict(list)
        }
        
        root = Path(root_path)
        
        for py_file in root.rglob("*.py"):
            # Skip test files and __pycache__
            if '__pycache__' in str(py_file) or 'test' in py_file.name.lower():
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                file_info = {
                    'path': str(py_file),
                    'relative_path': str(py_file.relative_to(root)),
                    'lines': len(lines),
                    'imports': self._extract_imports_quick(lines),
                    'classes': self._extract_classes_quick(lines),
                    'functions': self._extract_functions_quick(lines),
                    'module': self._get_module_name(py_file, root),
                    'hash': self._hash_file_content(''.join(lines[:100]))  # Hash first 100 lines
                }
                
                metadata['files'].append(file_info)
                metadata['total_lines'] += file_info['lines']
                metadata['total_files'] += 1
                metadata['file_types'][py_file.suffix] += 1
                metadata['module_structure'][file_info['module']].append(file_info['relative_path'])
                
                # Cache file info
                self.file_cache[file_info['path']] = file_info
                
            except Exception as e:
                print(f"Error scanning {py_file}: {e}")
                
        return metadata
    
    def _extract_imports_quick(self, lines: List[str]) -> List[str]:
        """Quickly extract imports without full AST parsing."""
        imports = []
        for line in lines[:50]:  # Check first 50 lines
            if line.strip().startswith(('import ', 'from ')):
                imports.append(line.strip())
        return imports
    
    def _extract_classes_quick(self, lines: List[str]) -> List[str]:
        """Quickly extract class names."""
        classes = []
        for line in lines:
            if line.strip().startswith('class '):
                match = re.match(r'class\s+(\w+)', line.strip())
                if match:
                    classes.append(match.group(1))
        return classes
    
    def _extract_functions_quick(self, lines: List[str]) -> List[str]:
        """Quickly extract function names."""
        functions = []
        for line in lines:
            if line.strip().startswith('def '):
                match = re.match(r'def\s+(\w+)', line.strip())
                if match:
                    functions.append(match.group(1))
        return functions
    
    def _get_module_name(self, file_path: Path, root: Path) -> str:
        """Get module name from file path."""
        relative = file_path.relative_to(root)
        parts = relative.parts[:-1]  # Exclude filename
        return '.'.join(parts) if parts else 'root'
    
    def _hash_file_content(self, content: str) -> str:
        """Generate hash of file content for comparison."""
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _create_functional_clusters(self, metadata: Dict[str, Any]) -> List[FunctionalCluster]:
        """
        Phase 2: Create functional clusters based on imports and structure.
        
        Groups related files into functional clusters for chunk-based analysis.
        """
        clusters = []
        processed_files = set()
        
        # First, cluster by module structure
        for module, files in metadata['module_structure'].items():
            if not files:
                continue
                
            cluster = FunctionalCluster(
                id=f"cluster_{module}",
                name=module,
                files=files,
                primary_purpose=self._infer_module_purpose(module, files),
                metrics=CodeMetrics()
            )
            
            # Calculate cluster metrics
            for file_path in files:
                if file_path in self.file_cache:
                    file_info = self.file_cache[file_path]
                    cluster.metrics.lines_of_code += file_info['lines']
                    processed_files.add(file_path)
            
            clusters.append(cluster)
        
        # Second, identify cross-cutting concerns
        cross_cutting = self._identify_cross_cutting_concerns(metadata)
        for concern_name, files in cross_cutting.items():
            cluster = FunctionalCluster(
                id=f"cluster_cross_{concern_name}",
                name=f"Cross-cutting: {concern_name}",
                files=files,
                primary_purpose=concern_name,
                metrics=CodeMetrics()
            )
            clusters.append(cluster)
        
        # Third, merge small clusters
        clusters = self._merge_small_clusters(clusters)
        
        return clusters
    
    def _infer_module_purpose(self, module_name: str, files: List[str]) -> str:
        """Infer the purpose of a module from its name and files."""
        purpose_keywords = {
            'api': 'API endpoints and interfaces',
            'model': 'Data models and schemas',
            'view': 'User interface components',
            'controller': 'Business logic controllers',
            'service': 'Service layer components',
            'util': 'Utility functions and helpers',
            'test': 'Test suites and fixtures',
            'config': 'Configuration management',
            'auth': 'Authentication and authorization',
            'db': 'Database operations',
            'cache': 'Caching mechanisms',
            'monitoring': 'System monitoring and metrics'
        }
        
        module_lower = module_name.lower()
        for keyword, purpose in purpose_keywords.items():
            if keyword in module_lower:
                return purpose
        
        # Check file names for hints
        file_purposes = defaultdict(int)
        for file_path in files:
            file_name = Path(file_path).stem.lower()
            for keyword, purpose in purpose_keywords.items():
                if keyword in file_name:
                    file_purposes[purpose] += 1
        
        if file_purposes:
            return max(file_purposes, key=file_purposes.get)
        
        return "General purpose module"
    
    def _identify_cross_cutting_concerns(self, metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify cross-cutting concerns like logging, security, etc."""
        concerns = defaultdict(list)
        
        concern_patterns = {
            'logging': ['logging', 'logger', 'log'],
            'security': ['auth', 'security', 'permission', 'crypto'],
            'caching': ['cache', 'redis', 'memcache'],
            'database': ['db', 'database', 'sql', 'orm'],
            'api': ['api', 'endpoint', 'route', 'rest'],
            'testing': ['test', 'mock', 'fixture', 'assert']
        }
        
        for file_info in metadata['files']:
            file_content_hints = ' '.join(file_info['imports'] + 
                                         file_info['classes'] + 
                                         file_info['functions']).lower()
            
            for concern, patterns in concern_patterns.items():
                if any(pattern in file_content_hints for pattern in patterns):
                    concerns[concern].append(file_info['relative_path'])
        
        # Filter out small concerns
        return {k: v for k, v in concerns.items() if len(v) >= 3}
    
    def _merge_small_clusters(self, clusters: List[FunctionalCluster], 
                            min_size: int = 500) -> List[FunctionalCluster]:
        """Merge small clusters to optimize analysis."""
        merged = []
        small_clusters = []
        
        for cluster in clusters:
            if cluster.metrics.lines_of_code < min_size:
                small_clusters.append(cluster)
            else:
                merged.append(cluster)
        
        # Merge small clusters by similarity
        if small_clusters:
            misc_cluster = FunctionalCluster(
                id="cluster_misc",
                name="Miscellaneous",
                files=[],
                primary_purpose="Mixed utility components",
                metrics=CodeMetrics()
            )
            
            for small in small_clusters:
                misc_cluster.files.extend(small.files)
                misc_cluster.metrics.lines_of_code += small.metrics.lines_of_code
                misc_cluster.sub_clusters.append(small)
            
            merged.append(misc_cluster)
        
        return merged
    
    def _analyze_clusters_progressively(self, clusters: List[FunctionalCluster]) -> List[FunctionalCluster]:
        """
        Phase 3: Progressive analysis of clusters.
        
        Analyze each cluster in detail, identifying refactoring opportunities.
        """
        analyzed = []
        futures = []
        
        # Submit cluster analysis tasks
        for cluster in clusters:
            future = self.executor.submit(self._analyze_single_cluster, cluster)
            futures.append((future, cluster))
        
        # Collect results
        for future, cluster in futures:
            try:
                analyzed_cluster = future.result(timeout=60)
                analyzed.append(analyzed_cluster)
            except Exception as e:
                print(f"Error analyzing cluster {cluster.name}: {e}")
                analyzed.append(cluster)
        
        return analyzed
    
    def _analyze_single_cluster(self, cluster: FunctionalCluster) -> FunctionalCluster:
        """Analyze a single cluster in detail."""
        print(f"Analyzing cluster: {cluster.name} ({cluster.metrics.lines_of_code} lines)")
        
        # Analyze each file in the cluster
        for file_path in cluster.files:
            if file_path in self.file_cache:
                file_info = self.file_cache[file_path]
                
                # For large files, analyze in chunks
                if file_info['lines'] > self.max_file_size:
                    opportunities = self._analyze_large_file_chunked(file_info)
                else:
                    opportunities = self._analyze_file_detailed(file_info)
                
                cluster.refactor_opportunities.extend(opportunities)
        
        # Identify cluster-level refactoring opportunities
        cluster_opportunities = self._identify_cluster_refactoring(cluster)
        cluster.refactor_opportunities.extend(cluster_opportunities)
        
        # Calculate cluster metrics
        cluster.metrics = self._calculate_cluster_metrics(cluster)
        
        return cluster
    
    def _analyze_file_detailed(self, file_info: Dict[str, Any]) -> List[RefactorOpportunity]:
        """Detailed analysis of a single file."""
        opportunities = []
        
        try:
            with open(file_info['path'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Check for various refactoring opportunities
            opportunities.extend(self._check_long_methods(tree, file_info['path']))
            opportunities.extend(self._check_complex_classes(tree, file_info['path']))
            opportunities.extend(self._check_code_duplication(tree, file_info['path']))
            opportunities.extend(self._check_god_objects(tree, file_info['path']))
            
        except Exception as e:
            print(f"Error analyzing {file_info['path']}: {e}")
        
        return opportunities
    
    def _analyze_large_file_chunked(self, file_info: Dict[str, Any]) -> List[RefactorOpportunity]:
        """Analyze large files in chunks."""
        opportunities = []
        
        # For very large files, suggest splitting
        opportunities.append(RefactorOpportunity(
            id=f"split_large_file_{file_info['path']}",
            type="split_file",
            severity="high",
            location=file_info['path'],
            description=f"File has {file_info['lines']} lines. Consider splitting into smaller modules.",
            estimated_effort=8,
            impact_score=85.0,
            dependencies=[]
        ))
        
        # Do lightweight analysis
        if file_info['classes'] and len(file_info['classes']) > 10:
            opportunities.append(RefactorOpportunity(
                id=f"too_many_classes_{file_info['path']}",
                type="split_module",
                severity="medium",
                location=file_info['path'],
                description=f"File contains {len(file_info['classes'])} classes. Consider module separation.",
                estimated_effort=6,
                impact_score=70.0,
                dependencies=[]
            ))
        
        return opportunities
    
    def _check_long_methods(self, tree: ast.AST, file_path: str) -> List[RefactorOpportunity]:
        """Check for long methods that should be refactored."""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count lines in function
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    lines = node.end_lineno - node.lineno
                    if lines > 50:
                        opportunities.append(RefactorOpportunity(
                            id=f"long_method_{file_path}_{node.name}",
                            type="extract_method",
                            severity="medium" if lines < 100 else "high",
                            location=f"{file_path}:{node.lineno}",
                            description=f"Method '{node.name}' has {lines} lines. Consider extracting smaller methods.",
                            estimated_effort=3,
                            impact_score=60.0,
                            dependencies=[]
                        ))
        
        return opportunities
    
    def _check_complex_classes(self, tree: ast.AST, file_path: str) -> List[RefactorOpportunity]:
        """Check for overly complex classes."""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                attributes = [n for n in node.body if isinstance(n, ast.Assign)]
                
                if len(methods) > 20:
                    opportunities.append(RefactorOpportunity(
                        id=f"complex_class_{file_path}_{node.name}",
                        type="split_class",
                        severity="high",
                        location=f"{file_path}:{node.lineno}",
                        description=f"Class '{node.name}' has {len(methods)} methods. Consider splitting responsibilities.",
                        estimated_effort=8,
                        impact_score=75.0,
                        dependencies=[]
                    ))
                
                if len(attributes) > 15:
                    opportunities.append(RefactorOpportunity(
                        id=f"too_many_attributes_{file_path}_{node.name}",
                        type="extract_data_class",
                        severity="medium",
                        location=f"{file_path}:{node.lineno}",
                        description=f"Class '{node.name}' has {len(attributes)} attributes. Consider data class extraction.",
                        estimated_effort=4,
                        impact_score=55.0,
                        dependencies=[]
                    ))
        
        return opportunities
    
    def _check_code_duplication(self, tree: ast.AST, file_path: str) -> List[RefactorOpportunity]:
        """Check for code duplication patterns."""
        opportunities = []
        
        # Simplified duplication check - in production, use more sophisticated algorithms
        function_bodies = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Convert function body to string for comparison
                body_str = ast.dump(node)
                if any(self._similarity_score(body_str, other) > 0.8 for other in function_bodies):
                    opportunities.append(RefactorOpportunity(
                        id=f"duplicate_code_{file_path}_{node.name}",
                        type="remove_duplication",
                        severity="medium",
                        location=f"{file_path}:{node.lineno}",
                        description=f"Function '{node.name}' contains duplicate code patterns.",
                        estimated_effort=2,
                        impact_score=50.0,
                        dependencies=[]
                    ))
                function_bodies.append(body_str)
        
        return opportunities
    
    def _check_god_objects(self, tree: ast.AST, file_path: str) -> List[RefactorOpportunity]:
        """Check for god objects (classes doing too much)."""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for god object patterns
                imports_in_methods = 0
                external_calls = 0
                
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        for subnode in ast.walk(method):
                            if isinstance(subnode, ast.Import):
                                imports_in_methods += 1
                            elif isinstance(subnode, ast.Call):
                                external_calls += 1
                
                if imports_in_methods > 5 or external_calls > 50:
                    opportunities.append(RefactorOpportunity(
                        id=f"god_object_{file_path}_{node.name}",
                        type="decompose_god_object",
                        severity="high",
                        location=f"{file_path}:{node.lineno}",
                        description=f"Class '{node.name}' appears to be a god object with too many responsibilities.",
                        estimated_effort=12,
                        impact_score=90.0,
                        dependencies=[]
                    ))
        
        return opportunities
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings (simplified)."""
        if str1 == str2:
            return 1.0
        
        # Simple character overlap ratio
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _identify_cluster_refactoring(self, cluster: FunctionalCluster) -> List[RefactorOpportunity]:
        """Identify refactoring opportunities at the cluster level."""
        opportunities = []
        
        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies(cluster)
        if circular_deps:
            opportunities.append(RefactorOpportunity(
                id=f"circular_deps_{cluster.id}",
                type="break_circular_dependency",
                severity="high",
                location=cluster.name,
                description=f"Cluster has {len(circular_deps)} circular dependencies.",
                estimated_effort=6,
                impact_score=80.0,
                dependencies=[]
            ))
        
        # Check for poor cohesion
        if cluster.metrics.cohesion_score < 0.5:
            opportunities.append(RefactorOpportunity(
                id=f"low_cohesion_{cluster.id}",
                type="improve_cohesion",
                severity="medium",
                location=cluster.name,
                description="Cluster has low cohesion. Consider reorganizing related functionality.",
                estimated_effort=8,
                impact_score=65.0,
                dependencies=[]
            ))
        
        # Check for high coupling
        if cluster.metrics.coupling_score > 0.7:
            opportunities.append(RefactorOpportunity(
                id=f"high_coupling_{cluster.id}",
                type="reduce_coupling",
                severity="medium",
                location=cluster.name,
                description="Cluster has high coupling. Consider introducing interfaces or abstractions.",
                estimated_effort=10,
                impact_score=70.0,
                dependencies=[]
            ))
        
        return opportunities
    
    def _find_circular_dependencies(self, cluster: FunctionalCluster) -> List[List[str]]:
        """Find circular dependencies within a cluster."""
        # Simplified circular dependency detection
        cycles = []
        
        # Build dependency graph for cluster
        cluster_deps = defaultdict(set)
        for file_path in cluster.files:
            if file_path in self.file_cache:
                file_info = self.file_cache[file_path]
                for imp in file_info['imports']:
                    # Extract imported module
                    if 'from' in imp:
                        parts = imp.split('from')[1].split('import')[0].strip()
                        cluster_deps[file_path].add(parts)
        
        # Detect cycles (simplified DFS)
        visited = set()
        for start_node in cluster_deps:
            if start_node not in visited:
                path = []
                if self._has_cycle_dfs(start_node, cluster_deps, visited, path, set()):
                    cycles.append(path)
        
        return cycles
    
    def _has_cycle_dfs(self, node: str, graph: Dict[str, Set[str]], 
                       visited: Set[str], path: List[str], rec_stack: Set[str]) -> bool:
        """DFS to detect cycles."""
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if self._has_cycle_dfs(neighbor, graph, visited, path, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True
        
        path.pop()
        rec_stack.remove(node)
        return False
    
    def _calculate_cluster_metrics(self, cluster: FunctionalCluster) -> CodeMetrics:
        """Calculate comprehensive metrics for a cluster."""
        metrics = CodeMetrics()
        
        # Aggregate basic metrics
        for file_path in cluster.files:
            if file_path in self.file_cache:
                file_info = self.file_cache[file_path]
                metrics.lines_of_code += file_info['lines']
                
                # Estimate complexity based on structure
                metrics.cyclomatic_complexity += len(file_info['functions']) * 3
                metrics.cognitive_complexity += len(file_info['classes']) * 5
        
        # Calculate cohesion (simplified - how related are the files)
        total_internal_deps = 0
        total_external_deps = 0
        
        for file_path in cluster.files:
            if file_path in self.file_cache:
                file_info = self.file_cache[file_path]
                for imp in file_info['imports']:
                    if any(f in imp for f in cluster.files):
                        total_internal_deps += 1
                    else:
                        total_external_deps += 1
        
        if total_internal_deps + total_external_deps > 0:
            metrics.cohesion_score = total_internal_deps / (total_internal_deps + total_external_deps)
            metrics.coupling_score = total_external_deps / (total_internal_deps + total_external_deps)
        
        return metrics
    
    def _build_hierarchy(self, root_path: str, metadata: Dict[str, Any], 
                        clusters: List[FunctionalCluster]) -> CodebaseHierarchy:
        """Build the final hierarchical representation."""
        
        # Calculate global metrics
        global_metrics = CodeMetrics(
            lines_of_code=metadata['total_lines'],
            cyclomatic_complexity=sum(c.metrics.cyclomatic_complexity for c in clusters),
            cognitive_complexity=sum(c.metrics.cognitive_complexity for c in clusters),
            coupling_score=sum(c.metrics.coupling_score for c in clusters) / len(clusters) if clusters else 0,
            cohesion_score=sum(c.metrics.cohesion_score for c in clusters) / len(clusters) if clusters else 0
        )
        
        # Generate summary
        summary = self._generate_codebase_summary(metadata, clusters, global_metrics)
        
        hierarchy = CodebaseHierarchy(
            root_path=root_path,
            total_lines=metadata['total_lines'],
            total_files=metadata['total_files'],
            clusters=clusters,
            global_metrics=global_metrics,
            summary=summary
        )
        
        return hierarchy
    
    def _generate_codebase_summary(self, metadata: Dict[str, Any], 
                                  clusters: List[FunctionalCluster],
                                  metrics: CodeMetrics) -> str:
        """Generate a high-level summary of the codebase."""
        
        summary_parts = [
            f"Codebase contains {metadata['total_files']} files with {metadata['total_lines']} lines of code.",
            f"Organized into {len(clusters)} functional clusters.",
            f"Average cohesion score: {metrics.cohesion_score:.2f}",
            f"Average coupling score: {metrics.coupling_score:.2f}",
        ]
        
        # Identify main concerns
        main_purposes = [c.primary_purpose for c in clusters[:5]]
        summary_parts.append(f"Main components: {', '.join(main_purposes)}")
        
        # Count total refactoring opportunities
        total_opportunities = sum(len(c.refactor_opportunities) for c in clusters)
        if total_opportunities > 0:
            high_severity = sum(1 for c in clusters 
                              for o in c.refactor_opportunities 
                              if o.severity == 'high')
            summary_parts.append(f"Found {total_opportunities} refactoring opportunities ({high_severity} high severity)")
        
        return " ".join(summary_parts)
    
    def generate_refactor_roadmap(self, hierarchy: CodebaseHierarchy = None) -> RefactorRoadmap:
        """
        Generate a comprehensive refactoring roadmap.
        
        Creates a phased plan for refactoring the codebase.
        """
        if hierarchy is None:
            hierarchy = self.current_hierarchy
            
        if not hierarchy:
            raise ValueError("No hierarchy available. Run analyze_codebase first.")
        
        # Collect all refactoring opportunities
        all_opportunities = []
        for cluster in hierarchy.clusters:
            all_opportunities.extend(cluster.refactor_opportunities)
        
        # Sort by impact and severity
        all_opportunities.sort(key=lambda x: (
            x.severity == 'high',
            x.impact_score,
            -x.estimated_effort
        ), reverse=True)
        
        # Create phases
        phases = self._create_refactoring_phases(all_opportunities)
        
        # Build dependency graph
        dep_graph = self._build_dependency_graph(all_opportunities)
        
        # Calculate total effort
        total_effort = sum(o.estimated_effort for o in all_opportunities)
        
        # Assess risk
        risk_assessment = self._assess_refactoring_risk(all_opportunities, hierarchy)
        
        roadmap = RefactorRoadmap(
            id=f"roadmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=f"Refactoring Roadmap for {Path(hierarchy.root_path).name}",
            phases=phases,
            total_effort_hours=total_effort,
            priority_score=self._calculate_priority_score(all_opportunities),
            dependencies_graph=dep_graph,
            risk_assessment=risk_assessment
        )
        
        return roadmap
    
    def _create_refactoring_phases(self, opportunities: List[RefactorOpportunity]) -> List[Dict[str, Any]]:
        """Create phased refactoring plan."""
        phases = []
        
        # Phase 1: Quick wins (low effort, high impact)
        quick_wins = [o for o in opportunities 
                     if o.estimated_effort <= 3 and o.impact_score >= 50]
        if quick_wins:
            phases.append({
                'phase': 1,
                'name': 'Quick Wins',
                'description': 'Low-effort, high-impact improvements',
                'opportunities': [asdict(o) for o in quick_wins[:10]],
                'estimated_duration': f"{sum(o.estimated_effort for o in quick_wins[:10])} hours"
            })
        
        # Phase 2: Critical issues (high severity)
        critical = [o for o in opportunities 
                   if o.severity == 'high' and o not in quick_wins]
        if critical:
            phases.append({
                'phase': 2,
                'name': 'Critical Refactoring',
                'description': 'Address high-severity issues',
                'opportunities': [asdict(o) for o in critical[:8]],
                'estimated_duration': f"{sum(o.estimated_effort for o in critical[:8])} hours"
            })
        
        # Phase 3: Structural improvements
        structural = [o for o in opportunities 
                     if o.type in ['split_class', 'split_file', 'decompose_god_object']
                     and o not in quick_wins and o not in critical]
        if structural:
            phases.append({
                'phase': 3,
                'name': 'Structural Improvements',
                'description': 'Major architectural refactoring',
                'opportunities': [asdict(o) for o in structural[:6]],
                'estimated_duration': f"{sum(o.estimated_effort for o in structural[:6])} hours"
            })
        
        # Phase 4: Optimization
        remaining = [o for o in opportunities 
                    if o not in quick_wins and o not in critical and o not in structural]
        if remaining:
            phases.append({
                'phase': 4,
                'name': 'Optimization',
                'description': 'Performance and maintainability improvements',
                'opportunities': [asdict(o) for o in remaining[:10]],
                'estimated_duration': f"{sum(o.estimated_effort for o in remaining[:10])} hours"
            })
        
        return phases
    
    def _build_dependency_graph(self, opportunities: List[RefactorOpportunity]) -> Dict[str, List[str]]:
        """Build dependency graph for refactoring tasks."""
        dep_graph = {}
        
        for opp in opportunities:
            dep_graph[opp.id] = opp.dependencies
            
            # Add implicit dependencies based on location
            for other in opportunities:
                if other.id != opp.id:
                    # If they affect the same file, create dependency
                    if opp.location in other.location or other.location in opp.location:
                        if other.severity == 'high' and opp.severity != 'high':
                            dep_graph[opp.id].append(other.id)
        
        return dep_graph
    
    def _calculate_priority_score(self, opportunities: List[RefactorOpportunity]) -> float:
        """Calculate overall priority score for the roadmap."""
        if not opportunities:
            return 0.0
        
        total_impact = sum(o.impact_score for o in opportunities)
        total_effort = sum(o.estimated_effort for o in opportunities)
        high_severity_count = sum(1 for o in opportunities if o.severity == 'high')
        
        # Priority formula: impact/effort ratio with severity weighting
        priority = (total_impact / max(total_effort, 1)) * (1 + high_severity_count * 0.1)
        
        return min(priority, 100.0)  # Cap at 100
    
    def _assess_refactoring_risk(self, opportunities: List[RefactorOpportunity], 
                                hierarchy: CodebaseHierarchy) -> str:
        """Assess the risk level of the refactoring effort."""
        
        risk_factors = []
        
        # Check for high coupling
        if hierarchy.global_metrics.coupling_score > 0.7:
            risk_factors.append("High coupling may complicate refactoring")
        
        # Check for lack of tests
        if hierarchy.global_metrics.test_coverage < 50:
            risk_factors.append("Low test coverage increases regression risk")
        
        # Check for large-scale changes
        major_changes = sum(1 for o in opportunities 
                          if o.type in ['split_file', 'decompose_god_object'])
        if major_changes > 5:
            risk_factors.append(f"{major_changes} major structural changes required")
        
        # Determine overall risk level
        if len(risk_factors) >= 3:
            risk_level = "HIGH"
        elif len(risk_factors) >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        risk_summary = f"Risk Level: {risk_level}. "
        if risk_factors:
            risk_summary += "Factors: " + "; ".join(risk_factors)
        else:
            risk_summary += "No significant risk factors identified."
        
        return risk_summary
    
    def export_analysis(self, output_path: str, hierarchy: CodebaseHierarchy = None):
        """Export the analysis results to a file."""
        if hierarchy is None:
            hierarchy = self.current_hierarchy
            
        if not hierarchy:
            raise ValueError("No analysis available to export")
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'root_path': hierarchy.root_path,
            'total_lines': hierarchy.total_lines,
            'total_files': hierarchy.total_files,
            'summary': hierarchy.summary,
            'global_metrics': asdict(hierarchy.global_metrics),
            'clusters': [
                {
                    'id': c.id,
                    'name': c.name,
                    'files_count': len(c.files),
                    'primary_purpose': c.primary_purpose,
                    'metrics': asdict(c.metrics),
                    'refactor_opportunities': [asdict(o) for o in c.refactor_opportunities[:5]]
                }
                for c in hierarchy.clusters
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Analysis exported to {output_path}")

# Factory function
def create_hierarchical_analyzer() -> HierarchicalRefactoringAnalyzer:
    """Create a hierarchical refactoring analyzer instance."""
    return HierarchicalRefactoringAnalyzer()