#!/usr/bin/env python3
"""
Cross-System Dependencies Analyzer - Agent D Hour 11
Comprehensive cross-system dependency analysis and validation
"""

import json
import time
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import ast
import re
import subprocess
from collections import defaultdict, deque
import hashlib
import yaml

@dataclass
class SystemDependency:
    """Represents a dependency between systems or components"""
    source: str
    target: str
    dependency_type: str  # "import", "api", "config", "data", "service", "file"
    strength: float  # 0.0 to 1.0 indicating dependency strength
    criticality: str  # "critical", "high", "medium", "low"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_validated: Optional[str] = None

@dataclass
class SystemComponent:
    """Represents a system component for dependency analysis"""
    name: str
    component_type: str  # "module", "service", "api", "database", "config"
    file_path: str
    dependencies: List[SystemDependency] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    size_metrics: Dict[str, int] = field(default_factory=dict)

@dataclass
class DependencyViolation:
    """Represents a dependency violation or issue"""
    violation_type: str  # "circular", "missing", "unused", "version_mismatch", "security"
    severity: str  # "critical", "high", "medium", "low"
    source: str
    target: str
    description: str
    suggested_fix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DependencyCluster:
    """Represents a cluster of related dependencies"""
    cluster_id: str
    components: List[str]
    cluster_type: str  # "tightly_coupled", "loosely_coupled", "isolated"
    cohesion_score: float  # 0.0 to 1.0
    coupling_score: float  # 0.0 to 1.0
    suggested_refactoring: str = ""

class CrossSystemDependencyAnalyzer:
    """Comprehensive cross-system dependency analysis and validation"""
    
    def __init__(self, base_path: Union[str, Path] = "."):
        self.base_path = Path(base_path)
        self.components: Dict[str, SystemComponent] = {}
        self.dependencies: List[SystemDependency] = []
        self.dependency_graph = nx.DiGraph()
        self.violations: List[DependencyViolation] = []
        self.clusters: List[DependencyCluster] = []
        self.config = self._load_dependency_config()
        self.start_time = time.time()
        
    def _load_dependency_config(self) -> Dict[str, Any]:
        """Load dependency analysis configuration"""
        default_config = {
            "analysis_depth": 3,  # levels of dependency traversal
            "ignore_patterns": [
                "__pycache__",
                ".git",
                ".venv",
                "node_modules", 
                ".pytest_cache",
                "test_backup",
                "archive"
            ],
            "dependency_weights": {
                "import": 0.8,
                "api": 0.9,
                "config": 0.6,
                "data": 0.7,
                "service": 0.9,
                "file": 0.5
            },
            "criticality_thresholds": {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.4,
                "low": 0.0
            },
            "clustering": {
                "min_cluster_size": 3,
                "cohesion_threshold": 0.6,
                "coupling_threshold": 0.3
            },
            "violation_detection": {
                "detect_circular": True,
                "detect_missing": True,
                "detect_unused": True,
                "detect_version_conflicts": True,
                "max_dependency_depth": 5
            }
        }
        
        config_file = self.base_path / "dependency_analysis_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def discover_system_components(self) -> Dict[str, SystemComponent]:
        """Discover all system components for dependency analysis"""
        print("Discovering system components...")
        components = {}
        
        # Discover Python modules
        for py_file in self.base_path.rglob("*.py"):
            if self._should_analyze_file(py_file):
                component = self._analyze_python_component(py_file)
                components[component.name] = component
        
        # Discover configuration components
        config_components = self._discover_config_components()
        components.update(config_components)
        
        # Discover API components
        api_components = self._discover_api_components()
        components.update(api_components)
        
        # Discover service components
        service_components = self._discover_service_components()
        components.update(service_components)
        
        self.components = components
        return components
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if file should be analyzed"""
        return not any(pattern in str(file_path) for pattern in self.config["ignore_patterns"])
    
    def _analyze_python_component(self, file_path: Path) -> SystemComponent:
        """Analyze Python component for dependencies"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            component_name = str(file_path.relative_to(self.base_path))
            dependencies = []
            interfaces = []
            exports = []
            
            # Extract imports (dependencies)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(SystemDependency(
                            source=component_name,
                            target=alias.name,
                            dependency_type="import",
                            strength=self.config["dependency_weights"]["import"],
                            criticality=self._determine_criticality(alias.name),
                            description=f"Import dependency on {alias.name}"
                        ))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    dependencies.append(SystemDependency(
                        source=component_name,
                        target=node.module,
                        dependency_type="import",
                        strength=self.config["dependency_weights"]["import"],
                        criticality=self._determine_criticality(node.module),
                        description=f"From-import dependency on {node.module}"
                    ))
            
            # Extract function and class definitions (exports)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):  # Public functions
                        exports.append(f"function:{node.name}")
                        interfaces.append(f"function:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    exports.append(f"class:{node.name}")
                    interfaces.append(f"class:{node.name}")
            
            # Calculate size metrics
            lines = content.splitlines()
            size_metrics = {
                "lines_of_code": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
                "total_lines": len(lines),
                "functions": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                "classes": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            }
            
            return SystemComponent(
                name=component_name,
                component_type="module",
                file_path=str(file_path),
                dependencies=dependencies,
                interfaces=interfaces,
                exports=exports,
                size_metrics=size_metrics
            )
            
        except Exception as e:
            return SystemComponent(
                name=str(file_path.relative_to(self.base_path)),
                component_type="module",
                file_path=str(file_path),
                size_metrics={"error": str(e)}
            )
    
    def _determine_criticality(self, dependency_name: str) -> str:
        """Determine criticality level of a dependency"""
        # Standard library dependencies are generally low criticality
        standard_libs = ["os", "sys", "json", "time", "datetime", "pathlib", "re", "asyncio", "subprocess"]
        if dependency_name in standard_libs:
            return "low"
        
        # Internal project dependencies are high criticality
        if dependency_name.startswith("TestMaster") or dependency_name.startswith("core"):
            return "high"
        
        # Third-party critical dependencies
        critical_deps = ["requests", "flask", "fastapi", "sqlalchemy", "pytest"]
        if any(dep in dependency_name.lower() for dep in critical_deps):
            return "critical"
        
        # Default to medium criticality
        return "medium"
    
    def _discover_config_components(self) -> Dict[str, SystemComponent]:
        """Discover configuration-based components"""
        components = {}
        
        for config_file in self.base_path.rglob("*.{yaml,yml,json,toml,ini}"):
            if self._should_analyze_file(config_file):
                component_name = f"config:{config_file.relative_to(self.base_path)}"
                
                # Analyze config dependencies
                dependencies = self._analyze_config_dependencies(config_file)
                
                components[component_name] = SystemComponent(
                    name=component_name,
                    component_type="config",
                    file_path=str(config_file),
                    dependencies=dependencies,
                    size_metrics={"file_size": config_file.stat().st_size}
                )
        
        return components
    
    def _analyze_config_dependencies(self, config_file: Path) -> List[SystemDependency]:
        """Analyze dependencies in configuration files"""
        dependencies = []
        
        try:
            if config_file.suffix in ['.yaml', '.yml']:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            elif config_file.suffix == '.json':
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
            else:
                return dependencies
            
            component_name = f"config:{config_file.relative_to(self.base_path)}"
            
            # Look for service references, database connections, etc.
            if isinstance(config_data, dict):
                for key, value in config_data.items():
                    if isinstance(value, str):
                        # Look for file references
                        if any(ext in value for ext in ['.py', '.yaml', '.json']):
                            dependencies.append(SystemDependency(
                                source=component_name,
                                target=value,
                                dependency_type="config",
                                strength=self.config["dependency_weights"]["config"],
                                criticality="medium",
                                description=f"Configuration reference to {value}"
                            ))
                        # Look for service URLs or database connections
                        elif any(pattern in key.lower() for pattern in ['url', 'host', 'service', 'db']):
                            dependencies.append(SystemDependency(
                                source=component_name,
                                target=f"service:{value}",
                                dependency_type="service",
                                strength=self.config["dependency_weights"]["service"],
                                criticality="high",
                                description=f"Service dependency on {value}"
                            ))
                            
        except Exception:
            pass
        
        return dependencies
    
    def _discover_api_components(self) -> Dict[str, SystemComponent]:
        """Discover API-based components"""
        components = {}
        
        # Look for API definition files
        api_files = []
        api_files.extend(self.base_path.rglob("*openapi*"))
        api_files.extend(self.base_path.rglob("*swagger*"))
        api_files.extend(self.base_path.rglob("*api*.yaml"))
        api_files.extend(self.base_path.rglob("*api*.json"))
        
        for api_file in api_files:
            if self._should_analyze_file(api_file):
                component_name = f"api:{api_file.relative_to(self.base_path)}"
                
                dependencies = self._analyze_api_dependencies(api_file)
                
                components[component_name] = SystemComponent(
                    name=component_name,
                    component_type="api",
                    file_path=str(api_file),
                    dependencies=dependencies,
                    size_metrics={"file_size": api_file.stat().st_size}
                )
        
        return components
    
    def _analyze_api_dependencies(self, api_file: Path) -> List[SystemDependency]:
        """Analyze dependencies in API specification files"""
        dependencies = []
        
        try:
            if api_file.suffix in ['.yaml', '.yml']:
                with open(api_file, 'r') as f:
                    api_spec = yaml.safe_load(f) or {}
            elif api_file.suffix == '.json':
                with open(api_file, 'r') as f:
                    api_spec = json.load(f)
            else:
                return dependencies
            
            component_name = f"api:{api_file.relative_to(self.base_path)}"
            
            # Analyze API dependencies
            if 'paths' in api_spec:
                for path, path_spec in api_spec['paths'].items():
                    for method, method_spec in path_spec.items():
                        if isinstance(method_spec, dict):
                            # API endpoint dependency
                            dependencies.append(SystemDependency(
                                source=component_name,
                                target=f"endpoint:{method.upper()}:{path}",
                                dependency_type="api",
                                strength=self.config["dependency_weights"]["api"],
                                criticality="high",
                                description=f"API endpoint {method.upper()} {path}"
                            ))
                            
        except Exception:
            pass
        
        return dependencies
    
    def _discover_service_components(self) -> Dict[str, SystemComponent]:
        """Discover service-based components"""
        components = {}
        
        # Look for service definition patterns in Python files
        for py_file in self.base_path.rglob("*.py"):
            if self._should_analyze_file(py_file):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for service patterns
                    if any(pattern in content.lower() for pattern in ['service', 'client', 'api', 'endpoint']):
                        component_name = f"service:{py_file.relative_to(self.base_path)}"
                        
                        dependencies = self._analyze_service_dependencies(content, component_name)
                        
                        if dependencies:  # Only add if has service dependencies
                            components[component_name] = SystemComponent(
                                name=component_name,
                                component_type="service",
                                file_path=str(py_file),
                                dependencies=dependencies,
                                size_metrics={"lines": len(content.splitlines())}
                            )
                            
                except Exception:
                    continue
        
        return components
    
    def _analyze_service_dependencies(self, content: str, component_name: str) -> List[SystemDependency]:
        """Analyze service dependencies in code"""
        dependencies = []
        
        # Look for HTTP requests and service calls
        service_patterns = [
            (r'requests\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]', "api"),
            (r'http[s]?://[^\s\'"]+', "service"),
            (r'@app\.route\([\'"]([^\'"]+)[\'"]', "api"),
            (r'@\w+\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]', "api")
        ]
        
        for pattern, dep_type in service_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    target = match[1] if len(match) > 1 else match[0]
                else:
                    target = match
                
                dependencies.append(SystemDependency(
                    source=component_name,
                    target=f"{dep_type}:{target}",
                    dependency_type=dep_type,
                    strength=self.config["dependency_weights"].get(dep_type, 0.5),
                    criticality="high",
                    description=f"{dep_type.title()} dependency on {target}"
                ))
        
        return dependencies
    
    def build_dependency_graph(self) -> nx.DiGraph:
        """Build comprehensive dependency graph"""
        print("Building dependency graph...")
        
        self.dependency_graph = nx.DiGraph()
        
        # Add all components as nodes
        for component_name, component in self.components.items():
            self.dependency_graph.add_node(
                component_name,
                component_type=component.component_type,
                size_metrics=component.size_metrics,
                interfaces=len(component.interfaces),
                exports=len(component.exports)
            )
        
        # Add dependencies as edges
        for component_name, component in self.components.items():
            for dependency in component.dependencies:
                # Add edge with dependency metadata
                self.dependency_graph.add_edge(
                    dependency.source,
                    dependency.target,
                    dependency_type=dependency.dependency_type,
                    strength=dependency.strength,
                    criticality=dependency.criticality,
                    description=dependency.description
                )
                
                # Track all dependencies
                self.dependencies.append(dependency)
        
        return self.dependency_graph
    
    def detect_dependency_violations(self) -> List[DependencyViolation]:
        """Detect various types of dependency violations"""
        print("Detecting dependency violations...")
        violations = []
        
        if self.config["violation_detection"]["detect_circular"]:
            violations.extend(self._detect_circular_dependencies())
        
        if self.config["violation_detection"]["detect_missing"]:
            violations.extend(self._detect_missing_dependencies())
        
        if self.config["violation_detection"]["detect_unused"]:
            violations.extend(self._detect_unused_dependencies())
        
        violations.extend(self._detect_high_coupling())
        violations.extend(self._detect_deep_dependency_chains())
        
        self.violations = violations
        return violations
    
    def _detect_circular_dependencies(self) -> List[DependencyViolation]:
        """Detect circular dependency violations"""
        violations = []
        
        try:
            # Find strongly connected components
            strongly_connected = list(nx.strongly_connected_components(self.dependency_graph))
            
            for component_set in strongly_connected:
                if len(component_set) > 1:  # Circular dependency
                    components = list(component_set)
                    violations.append(DependencyViolation(
                        violation_type="circular",
                        severity="high",
                        source=components[0],
                        target=components[1] if len(components) > 1 else components[0],
                        description=f"Circular dependency detected among: {', '.join(components)}",
                        suggested_fix="Consider dependency injection or interface segregation",
                        metadata={"components": components, "cycle_length": len(components)}
                    ))
                    
        except Exception as e:
            violations.append(DependencyViolation(
                violation_type="analysis_error",
                severity="medium",
                source="analyzer",
                target="circular_detection",
                description=f"Error detecting circular dependencies: {str(e)}"
            ))
        
        return violations
    
    def _detect_missing_dependencies(self) -> List[DependencyViolation]:
        """Detect missing or broken dependencies"""
        violations = []
        
        for dependency in self.dependencies:
            # Check if target exists in our component map
            if (dependency.target not in self.components and 
                not dependency.target.startswith(('os', 'sys', 'json', 'time', 'datetime', 'pathlib', 're', 'asyncio'))):
                
                # Skip external services and APIs
                if not any(prefix in dependency.target for prefix in ['http', 'service:', 'api:', 'endpoint:']):
                    violations.append(DependencyViolation(
                        violation_type="missing",
                        severity="medium",
                        source=dependency.source,
                        target=dependency.target,
                        description=f"Missing dependency: {dependency.target} not found",
                        suggested_fix="Install missing package or fix import path",
                        metadata={"dependency_type": dependency.dependency_type}
                    ))
        
        return violations
    
    def _detect_unused_dependencies(self) -> List[DependencyViolation]:
        """Detect unused or redundant dependencies"""
        violations = []
        
        # Find components with no incoming dependencies (potential unused components)
        for component_name, component in self.components.items():
            if component.component_type == "module":
                incoming_deps = [dep for dep in self.dependencies if dep.target == component_name]
                if not incoming_deps and not component.exports:
                    violations.append(DependencyViolation(
                        violation_type="unused",
                        severity="low",
                        source=component_name,
                        target="none",
                        description=f"Potentially unused component: {component_name}",
                        suggested_fix="Consider removing if truly unused",
                        metadata={"size_metrics": component.size_metrics}
                    ))
        
        return violations
    
    def _detect_high_coupling(self) -> List[DependencyViolation]:
        """Detect components with high coupling"""
        violations = []
        
        for component_name, component in self.components.items():
            dependency_count = len(component.dependencies)
            dependent_count = len([dep for dep in self.dependencies if dep.target == component_name])
            
            total_coupling = dependency_count + dependent_count
            
            if total_coupling > 10:  # High coupling threshold
                violations.append(DependencyViolation(
                    violation_type="high_coupling",
                    severity="medium",
                    source=component_name,
                    target="multiple",
                    description=f"High coupling detected: {total_coupling} total dependencies",
                    suggested_fix="Consider refactoring to reduce dependencies",
                    metadata={
                        "outgoing_dependencies": dependency_count,
                        "incoming_dependencies": dependent_count,
                        "total_coupling": total_coupling
                    }
                ))
        
        return violations
    
    def _detect_deep_dependency_chains(self) -> List[DependencyViolation]:
        """Detect deep dependency chains"""
        violations = []
        
        max_depth = self.config["violation_detection"]["max_dependency_depth"]
        
        # Find longest paths in the dependency graph
        for component in self.components:
            try:
                # Calculate maximum depth from this component
                depths = nx.single_source_shortest_path_length(self.dependency_graph, component)
                max_component_depth = max(depths.values()) if depths else 0
                
                if max_component_depth > max_depth:
                    violations.append(DependencyViolation(
                        violation_type="deep_dependency_chain",
                        severity="medium",
                        source=component,
                        target="multiple",
                        description=f"Deep dependency chain detected: depth {max_component_depth}",
                        suggested_fix="Consider flattening dependency hierarchy",
                        metadata={"max_depth": max_component_depth}
                    ))
                    
            except Exception:
                continue
        
        return violations
    
    def analyze_dependency_clusters(self) -> List[DependencyCluster]:
        """Analyze dependency clustering and cohesion"""
        print("Analyzing dependency clusters...")
        clusters = []
        
        try:
            # Use community detection algorithms
            if len(self.dependency_graph) > 0:
                # Convert to undirected for community detection
                undirected_graph = self.dependency_graph.to_undirected()
                
                # Simple clustering based on connected components
                connected_components = list(nx.connected_components(undirected_graph))
                
                for i, component_set in enumerate(connected_components):
                    if len(component_set) >= self.config["clustering"]["min_cluster_size"]:
                        components = list(component_set)
                        
                        # Calculate cohesion and coupling metrics
                        cohesion_score = self._calculate_cohesion(components)
                        coupling_score = self._calculate_coupling(components)
                        
                        # Determine cluster type
                        cluster_type = "isolated"
                        if cohesion_score > self.config["clustering"]["cohesion_threshold"]:
                            cluster_type = "tightly_coupled"
                        elif coupling_score < self.config["clustering"]["coupling_threshold"]:
                            cluster_type = "loosely_coupled"
                        
                        # Generate refactoring suggestions
                        suggested_refactoring = self._generate_cluster_refactoring_suggestion(
                            components, cluster_type, cohesion_score, coupling_score
                        )
                        
                        clusters.append(DependencyCluster(
                            cluster_id=f"cluster_{i}",
                            components=components,
                            cluster_type=cluster_type,
                            cohesion_score=cohesion_score,
                            coupling_score=coupling_score,
                            suggested_refactoring=suggested_refactoring
                        ))
                        
        except Exception as e:
            print(f"Error in cluster analysis: {e}")
        
        self.clusters = clusters
        return clusters
    
    def _calculate_cohesion(self, components: List[str]) -> float:
        """Calculate cohesion score for a cluster of components"""
        if len(components) < 2:
            return 1.0
        
        internal_edges = 0
        total_possible_edges = len(components) * (len(components) - 1)
        
        for source in components:
            for target in components:
                if source != target and self.dependency_graph.has_edge(source, target):
                    internal_edges += 1
        
        return internal_edges / total_possible_edges if total_possible_edges > 0 else 0.0
    
    def _calculate_coupling(self, components: List[str]) -> float:
        """Calculate coupling score for a cluster of components"""
        external_edges = 0
        total_edges = 0
        
        for component in components:
            # Count outgoing edges
            for target in self.dependency_graph.successors(component):
                total_edges += 1
                if target not in components:
                    external_edges += 1
            
            # Count incoming edges
            for source in self.dependency_graph.predecessors(component):
                total_edges += 1
                if source not in components:
                    external_edges += 1
        
        return external_edges / total_edges if total_edges > 0 else 0.0
    
    def _generate_cluster_refactoring_suggestion(self, components: List[str], cluster_type: str,
                                               cohesion_score: float, coupling_score: float) -> str:
        """Generate refactoring suggestions for a cluster"""
        if cluster_type == "tightly_coupled" and coupling_score > 0.7:
            return "Consider extracting common functionality into a shared module"
        elif cluster_type == "loosely_coupled" and cohesion_score < 0.3:
            return "Consider separating into independent modules"
        elif coupling_score > 0.8:
            return "High external coupling detected - consider interface segregation"
        else:
            return "Well-structured cluster - no immediate refactoring needed"
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive cross-system dependency analysis report"""
        print("Generating comprehensive dependency analysis report...")
        
        # Discover components and build graph
        components = self.discover_system_components()
        dependency_graph = self.build_dependency_graph()
        violations = self.detect_dependency_violations()
        clusters = self.analyze_dependency_clusters()
        
        # Calculate summary statistics
        total_execution_time = time.time() - self.start_time
        
        # Component analysis summary
        component_summary = {
            "total_components": len(components),
            "components_by_type": self._count_components_by_type(components),
            "total_dependencies": len(self.dependencies),
            "dependencies_by_type": self._count_dependencies_by_type(),
            "average_dependencies_per_component": len(self.dependencies) / len(components) if components else 0
        }
        
        # Graph analysis summary
        graph_metrics = self._calculate_graph_metrics(dependency_graph)
        
        # Violation analysis summary
        violation_summary = {
            "total_violations": len(violations),
            "violations_by_type": self._count_violations_by_type(violations),
            "violations_by_severity": self._count_violations_by_severity(violations)
        }
        
        # Cluster analysis summary
        cluster_summary = {
            "total_clusters": len(clusters),
            "clusters_by_type": self._count_clusters_by_type(clusters),
            "average_cluster_size": sum(len(c.components) for c in clusters) / len(clusters) if clusters else 0,
            "average_cohesion": sum(c.cohesion_score for c in clusters) / len(clusters) if clusters else 0,
            "average_coupling": sum(c.coupling_score for c in clusters) / len(clusters) if clusters else 0
        }
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "execution_time": total_execution_time,
            "component_analysis": {
                "summary": component_summary,
                "components": {
                    name: {
                        "component_type": comp.component_type,
                        "file_path": comp.file_path,
                        "dependencies_count": len(comp.dependencies),
                        "interfaces_count": len(comp.interfaces),
                        "exports_count": len(comp.exports),
                        "size_metrics": comp.size_metrics
                    }
                    for name, comp in components.items()
                }
            },
            "dependency_analysis": {
                "summary": {
                    "total_dependencies": len(self.dependencies),
                    "dependency_types": self._count_dependencies_by_type(),
                    "criticality_distribution": self._count_dependencies_by_criticality()
                },
                "dependencies": [
                    {
                        "source": dep.source,
                        "target": dep.target,
                        "type": dep.dependency_type,
                        "strength": dep.strength,
                        "criticality": dep.criticality,
                        "description": dep.description
                    }
                    for dep in self.dependencies
                ]
            },
            "graph_analysis": {
                "metrics": graph_metrics,
                "summary": f"Dependency graph with {dependency_graph.number_of_nodes()} nodes and {dependency_graph.number_of_edges()} edges"
            },
            "violation_analysis": {
                "summary": violation_summary,
                "violations": [
                    {
                        "type": v.violation_type,
                        "severity": v.severity,
                        "source": v.source,
                        "target": v.target,
                        "description": v.description,
                        "suggested_fix": v.suggested_fix,
                        "metadata": v.metadata
                    }
                    for v in violations
                ]
            },
            "cluster_analysis": {
                "summary": cluster_summary,
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "components": c.components,
                        "cluster_type": c.cluster_type,
                        "cohesion_score": c.cohesion_score,
                        "coupling_score": c.coupling_score,
                        "suggested_refactoring": c.suggested_refactoring
                    }
                    for c in clusters
                ]
            },
            "recommendations": self._generate_recommendations(),
            "config_used": self.config
        }
        
        return report
    
    def _count_components_by_type(self, components: Dict[str, SystemComponent]) -> Dict[str, int]:
        """Count components by type"""
        type_counts = {}
        for component in components.values():
            comp_type = component.component_type
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        return type_counts
    
    def _count_dependencies_by_type(self) -> Dict[str, int]:
        """Count dependencies by type"""
        type_counts = {}
        for dependency in self.dependencies:
            dep_type = dependency.dependency_type
            type_counts[dep_type] = type_counts.get(dep_type, 0) + 1
        return type_counts
    
    def _count_dependencies_by_criticality(self) -> Dict[str, int]:
        """Count dependencies by criticality"""
        criticality_counts = {}
        for dependency in self.dependencies:
            criticality = dependency.criticality
            criticality_counts[criticality] = criticality_counts.get(criticality, 0) + 1
        return criticality_counts
    
    def _count_violations_by_type(self, violations: List[DependencyViolation]) -> Dict[str, int]:
        """Count violations by type"""
        type_counts = {}
        for violation in violations:
            v_type = violation.violation_type
            type_counts[v_type] = type_counts.get(v_type, 0) + 1
        return type_counts
    
    def _count_violations_by_severity(self, violations: List[DependencyViolation]) -> Dict[str, int]:
        """Count violations by severity"""
        severity_counts = {}
        for violation in violations:
            severity = violation.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts
    
    def _count_clusters_by_type(self, clusters: List[DependencyCluster]) -> Dict[str, int]:
        """Count clusters by type"""
        type_counts = {}
        for cluster in clusters:
            cluster_type = cluster.cluster_type
            type_counts[cluster_type] = type_counts.get(cluster_type, 0) + 1
        return type_counts
    
    def _calculate_graph_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate graph analysis metrics"""
        if len(graph) == 0:
            return {"nodes": 0, "edges": 0, "density": 0, "avg_degree": 0}
        
        metrics = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "avg_in_degree": sum(dict(graph.in_degree()).values()) / len(graph),
            "avg_out_degree": sum(dict(graph.out_degree()).values()) / len(graph)
        }
        
        # Additional metrics if graph is not empty
        try:
            if nx.is_weakly_connected(graph):
                metrics["weakly_connected"] = True
                metrics["diameter"] = nx.diameter(graph.to_undirected())
            else:
                metrics["weakly_connected"] = False
                
            metrics["number_of_weakly_connected_components"] = nx.number_weakly_connected_components(graph)
            metrics["number_of_strongly_connected_components"] = nx.number_strongly_connected_components(graph)
            
        except Exception:
            pass
        
        return metrics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Violation-based recommendations
        if self.violations:
            circular_violations = [v for v in self.violations if v.violation_type == "circular"]
            if circular_violations:
                recommendations.append(f"Resolve {len(circular_violations)} circular dependencies to improve maintainability")
            
            missing_violations = [v for v in self.violations if v.violation_type == "missing"]
            if missing_violations:
                recommendations.append(f"Fix {len(missing_violations)} missing dependencies")
            
            high_coupling_violations = [v for v in self.violations if v.violation_type == "high_coupling"]
            if high_coupling_violations:
                recommendations.append(f"Refactor {len(high_coupling_violations)} highly coupled components")
        
        # Cluster-based recommendations
        if self.clusters:
            tightly_coupled_clusters = [c for c in self.clusters if c.cluster_type == "tightly_coupled"]
            if tightly_coupled_clusters:
                recommendations.append(f"Consider refactoring {len(tightly_coupled_clusters)} tightly coupled clusters")
        
        # Graph-based recommendations
        if len(self.dependency_graph) > 0:
            avg_dependencies = len(self.dependencies) / len(self.components)
            if avg_dependencies > 5:
                recommendations.append("High average dependency count suggests complex architecture - consider simplification")
        
        if not recommendations:
            recommendations.append("Dependency analysis completed - no critical issues detected")
        
        return recommendations


def main():
    """Main execution function"""
    print("=== TestMaster Cross-System Dependencies Analyzer ===")
    print("Agent D - Hour 11: Cross-System Dependencies Analysis")
    print()
    
    # Initialize analyzer
    analyzer = CrossSystemDependencyAnalyzer()
    
    # Generate comprehensive report
    print("Phase 1: Cross-System Dependency Analysis")
    report = analyzer.generate_comprehensive_report()
    
    # Save report
    report_file = Path("TestMaster/docs/validation/cross_system_dependencies_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    print(f"\nCross-System Dependencies Analysis Complete!")
    print(f"Components: {report['component_analysis']['summary']['total_components']}")
    print(f"Dependencies: {report['dependency_analysis']['summary']['total_dependencies']}")
    print(f"Violations: {report['violation_analysis']['summary']['total_violations']}")
    print(f"Clusters: {report['cluster_analysis']['summary']['total_clusters']}")
    print(f"Execution Time: {report['execution_time']:.2f}s")
    print(f"\nReport saved: {report_file}")
    
    # Show recommendations
    if report['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")


if __name__ == "__main__":
    main()