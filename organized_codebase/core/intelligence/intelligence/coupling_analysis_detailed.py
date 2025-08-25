#!/usr/bin/env python3
"""
TestMaster Coupling Analysis - Agent B Phase 2 Hours 31-35
==========================================================

Detailed coupling and cohesion analysis for the TestMaster intelligence
framework. Analyzes module relationships using Martin's metrics and
identifies optimization opportunities.

Author: Agent B - Documentation & Modularization Excellence
Phase: 2 - Advanced Interdependency Analysis (Hours 31-35)
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
import json
import re


@dataclass
class CouplingMetrics:
    """Coupling metrics for a module using Martin's principles."""
    module_id: str
    afferent_coupling: int = 0  # Ca - modules that depend on this module
    efferent_coupling: int = 0  # Ce - modules this module depends on
    instability: float = 0.0    # I = Ce / (Ca + Ce)
    abstractness: float = 0.0   # A = abstract classes / total classes
    distance: float = 0.0       # D = |A + I - 1|
    cohesion_score: float = 0.0
    complexity_factor: float = 0.0
    interface_quality: float = 0.0


@dataclass
class InterfaceAnalysis:
    """Analysis of module interface quality and design."""
    module_id: str
    public_functions: List[str] = field(default_factory=list)
    public_classes: List[str] = field(default_factory=list)
    interface_complexity: float = 0.0
    parameter_consistency: float = 0.0
    return_type_consistency: float = 0.0
    documentation_coverage: float = 0.0
    stability_score: float = 0.0


class DetailedCouplingAnalyzer:
    """
    Detailed coupling and cohesion analyzer for TestMaster framework.
    
    Implements Martin's software metrics for dependency analysis and
    identifies areas for architectural improvement.
    """
    
    def __init__(self, root_path: str = "."):
        """Initialize coupling analyzer."""
        self.root_path = Path(root_path)
        self.modules = {}
        self.coupling_metrics = {}
        self.interface_analyses = {}
        self.dependency_matrix = {}
        
        # Key intelligence modules to analyze
        self.intelligence_modules = [
            "core/intelligence/__init__.py",
            "core/intelligence/testing/__init__.py",
            "core/intelligence/api/__init__.py", 
            "core/intelligence/analytics/__init__.py",
            "testmaster_orchestrator.py",
            "intelligent_test_builder.py",
            "enhanced_self_healing_verifier.py",
            "agentic_test_monitor.py",
            "parallel_converter.py",
            "config/__init__.py"
        ]
    
    def analyze_coupling_detailed(self) -> Dict[str, Any]:
        """
        Perform detailed coupling and cohesion analysis.
        
        Returns:
            Comprehensive coupling analysis results
        """
        print("Analyzing detailed coupling and cohesion metrics...")
        
        # Step 1: Discover and analyze key modules
        self._discover_key_modules()
        
        # Step 2: Build dependency matrix
        self._build_dependency_matrix()
        
        # Step 3: Calculate Martin's metrics
        self._calculate_martin_metrics()
        
        # Step 4: Analyze interfaces
        self._analyze_interfaces()
        
        # Step 5: Calculate cohesion metrics
        self._calculate_cohesion_metrics()
        
        # Step 6: Identify coupling issues
        coupling_issues = self._identify_coupling_issues()
        
        # Step 7: Generate optimization recommendations
        recommendations = self._generate_recommendations()
        
        results = {
            'analysis_metadata': {
                'analyzer': 'Agent B - Detailed Coupling Analysis',
                'phase': 'Hours 31-35',
                'modules_analyzed': len(self.coupling_metrics),
                'intelligence_modules': len(self.intelligence_modules)
            },
            'coupling_metrics': self.coupling_metrics,
            'interface_analyses': self.interface_analyses,
            'dependency_matrix': self.dependency_matrix,
            'coupling_issues': coupling_issues,
            'recommendations': recommendations,
            'summary_statistics': self._calculate_summary_statistics()
        }
        
        return results
    
    def _discover_key_modules(self):
        """Discover and analyze key framework modules."""
        print("Discovering key framework modules...")
        
        for module_path in self.intelligence_modules:
            full_path = self.root_path / module_path
            if full_path.exists():
                self.modules[module_path] = {
                    'path': str(full_path),
                    'exists': True,
                    'size': self._get_file_size(full_path),
                    'complexity': self._calculate_complexity(full_path)
                }
            else:
                self.modules[module_path] = {
                    'path': str(full_path),
                    'exists': False
                }
        
        print(f"Analyzed {len([m for m in self.modules.values() if m['exists']])} modules")
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in lines."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except Exception:
            return 0
    
    def _calculate_complexity(self, file_path: Path) -> float:
        """Calculate cyclomatic complexity."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            complexity = 1
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                   ast.ExceptHandler, ast.With, ast.AsyncWith)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except Exception:
            return 1.0
    
    def _build_dependency_matrix(self):
        """Build dependency matrix between modules."""
        print("Building dependency matrix...")
        
        for module_id, module_info in self.modules.items():
            if not module_info['exists']:
                continue
                
            dependencies = self._extract_dependencies(Path(module_info['path']))
            self.dependency_matrix[module_id] = dependencies
    
    def _extract_dependencies(self, file_path: Path) -> List[str]:
        """Extract dependencies from a Python file."""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
        
        return dependencies
    
    def _calculate_martin_metrics(self):
        """Calculate Martin's coupling metrics for each module."""
        print("Calculating Martin's coupling metrics...")
        
        # Calculate afferent and efferent coupling
        for module_id in self.modules:
            if not self.modules[module_id]['exists']:
                continue
                
            ca = self._calculate_afferent_coupling(module_id)
            ce = self._calculate_efferent_coupling(module_id)
            
            # Calculate instability
            instability = ce / (ca + ce) if (ca + ce) > 0 else 0
            
            # Calculate abstractness
            abstractness = self._calculate_abstractness(module_id)
            
            # Calculate distance from main sequence
            distance = abs(abstractness + instability - 1)
            
            self.coupling_metrics[module_id] = CouplingMetrics(
                module_id=module_id,
                afferent_coupling=ca,
                efferent_coupling=ce,
                instability=instability,
                abstractness=abstractness,
                distance=distance
            )
    
    def _calculate_afferent_coupling(self, target_module: str) -> int:
        """Calculate afferent coupling (modules that depend on target)."""
        ca = 0
        target_patterns = [
            target_module.replace('.py', '').replace('/', '.'),
            target_module.replace('.py', '').replace('\\', '.'),
            target_module.split('/')[-1].replace('.py', ''),
            target_module.split('/')[-1].replace('__init__.py', '').rstrip('/')
        ]
        
        for module_id, deps in self.dependency_matrix.items():
            if module_id == target_module:
                continue
                
            for dep in deps:
                for pattern in target_patterns:
                    if pattern in dep or dep in pattern:
                        ca += 1
                        break
        
        return ca
    
    def _calculate_efferent_coupling(self, source_module: str) -> int:
        """Calculate efferent coupling (modules that source depends on)."""
        if source_module in self.dependency_matrix:
            return len(self.dependency_matrix[source_module])
        return 0
    
    def _calculate_abstractness(self, module_id: str) -> float:
        """Calculate abstractness ratio (abstract classes / total classes)."""
        if not self.modules[module_id]['exists']:
            return 0.0
            
        try:
            file_path = Path(self.modules[module_id]['path'])
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            total_classes = 0
            abstract_classes = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    total_classes += 1
                    
                    # Check if class has abstract methods
                    for class_node in node.body:
                        if (isinstance(class_node, ast.FunctionDef) and
                            any(decorator.id == 'abstractmethod' 
                                for decorator in class_node.decorator_list
                                if isinstance(decorator, ast.Name))):
                            abstract_classes += 1
                            break
            
            return abstract_classes / total_classes if total_classes > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_interfaces(self):
        """Analyze interface quality for each module."""
        print("Analyzing module interfaces...")
        
        for module_id, module_info in self.modules.items():
            if not module_info['exists']:
                continue
                
            interface = self._extract_interface(Path(module_info['path']))
            self.interface_analyses[module_id] = interface
    
    def _extract_interface(self, file_path: Path) -> InterfaceAnalysis:
        """Extract and analyze module interface."""
        interface = InterfaceAnalysis(module_id=str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract public functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):
                        interface.public_functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    if not node.name.startswith('_'):
                        interface.public_classes.append(node.name)
            
            # Calculate interface metrics
            interface.interface_complexity = len(interface.public_functions) + len(interface.public_classes)
            interface.documentation_coverage = self._calculate_interface_docs(content)
            interface.stability_score = 1.0 - self.coupling_metrics.get(
                str(file_path), CouplingMetrics("")
            ).instability
            
        except Exception as e:
            print(f"Warning: Could not analyze interface for {file_path}: {e}")
        
        return interface
    
    def _calculate_interface_docs(self, content: str) -> float:
        """Calculate documentation coverage for interface."""
        docstring_count = content.count('"""') // 2
        function_count = content.count('def ')
        class_count = content.count('class ')
        
        total_items = function_count + class_count
        return min(1.0, docstring_count / total_items) if total_items > 0 else 0.0
    
    def _calculate_cohesion_metrics(self):
        """Calculate cohesion metrics for modules."""
        print("Calculating cohesion metrics...")
        
        for module_id, metrics in self.coupling_metrics.items():
            if not self.modules[module_id]['exists']:
                continue
            
            # Calculate LCOM (Lack of Cohesion in Methods)
            cohesion = self._calculate_lcom(Path(self.modules[module_id]['path']))
            metrics.cohesion_score = 1.0 - cohesion  # Invert so higher is better
            
            # Calculate complexity factor
            complexity = self.modules[module_id]['complexity']
            size = self.modules[module_id]['size']
            metrics.complexity_factor = complexity / size if size > 0 else 0
            
            # Calculate interface quality
            interface = self.interface_analyses.get(module_id)
            if interface:
                metrics.interface_quality = (
                    interface.documentation_coverage * 0.4 +
                    (1.0 - min(1.0, interface.interface_complexity / 20)) * 0.3 +
                    interface.stability_score * 0.3
                )
    
    def _calculate_lcom(self, file_path: Path) -> float:
        """Calculate Lack of Cohesion in Methods (LCOM)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Find classes and analyze method cohesion
            total_lcom = 0.0
            class_count = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_count += 1
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    
                    if len(methods) <= 1:
                        continue
                    
                    # Calculate shared variable usage between methods
                    shared_vars = 0
                    total_pairs = 0
                    
                    for i, method1 in enumerate(methods):
                        vars1 = self._extract_variables(method1)
                        for j, method2 in enumerate(methods[i+1:], i+1):
                            vars2 = self._extract_variables(method2)
                            total_pairs += 1
                            if vars1 & vars2:  # Intersection of variables
                                shared_vars += 1
                    
                    if total_pairs > 0:
                        class_lcom = 1.0 - (shared_vars / total_pairs)
                        total_lcom += class_lcom
            
            return total_lcom / class_count if class_count > 0 else 0.0
            
        except Exception:
            return 0.5  # Default moderate cohesion
    
    def _extract_variables(self, method_node: ast.FunctionDef) -> Set[str]:
        """Extract variables used in a method."""
        variables = set()
        
        for node in ast.walk(method_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                variables.add(node.id)
            elif isinstance(node, ast.Attribute):
                variables.add(node.attr)
        
        return variables
    
    def _identify_coupling_issues(self) -> List[Dict[str, Any]]:
        """Identify modules with coupling issues."""
        issues = []
        
        for module_id, metrics in self.coupling_metrics.items():
            # High instability issues
            if metrics.instability > 0.8:
                issues.append({
                    'module': module_id,
                    'type': 'high_instability',
                    'severity': 'medium',
                    'value': metrics.instability,
                    'description': f"Module has high instability ({metrics.instability:.2f})"
                })
            
            # High distance from main sequence
            if metrics.distance > 0.7:
                issues.append({
                    'module': module_id,
                    'type': 'high_distance',
                    'severity': 'high',
                    'value': metrics.distance,
                    'description': f"Module is far from main sequence ({metrics.distance:.2f})"
                })
            
            # Low cohesion
            if metrics.cohesion_score < 0.3:
                issues.append({
                    'module': module_id,
                    'type': 'low_cohesion',
                    'severity': 'medium',
                    'value': metrics.cohesion_score,
                    'description': f"Module has low cohesion ({metrics.cohesion_score:.2f})"
                })
            
            # High complexity
            if metrics.complexity_factor > 0.1:
                issues.append({
                    'module': module_id,
                    'type': 'high_complexity',
                    'severity': 'low',
                    'value': metrics.complexity_factor,
                    'description': f"Module has high complexity factor ({metrics.complexity_factor:.2f})"
                })
        
        return issues
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze coupling patterns
        high_coupling_modules = [
            m for m, metrics in self.coupling_metrics.items()
            if metrics.instability > 0.7 or metrics.distance > 0.6
        ]
        
        if high_coupling_modules:
            recommendations.append({
                'category': 'coupling_optimization',
                'priority': 'high',
                'modules': high_coupling_modules,
                'description': 'Reduce coupling in high-instability modules',
                'actions': [
                    'Extract interfaces to reduce dependencies',
                    'Apply dependency inversion principle',
                    'Consider facade pattern for complex interfaces'
                ]
            })
        
        # Analyze cohesion patterns
        low_cohesion_modules = [
            m for m, metrics in self.coupling_metrics.items()
            if metrics.cohesion_score < 0.4
        ]
        
        if low_cohesion_modules:
            recommendations.append({
                'category': 'cohesion_improvement',
                'priority': 'medium',
                'modules': low_cohesion_modules,
                'description': 'Improve cohesion in loosely related modules',
                'actions': [
                    'Split modules with multiple responsibilities',
                    'Group related functionality together',
                    'Extract utility functions to shared modules'
                ]
            })
        
        # Interface quality recommendations
        poor_interface_modules = [
            m for m, metrics in self.coupling_metrics.items()
            if metrics.interface_quality < 0.5
        ]
        
        if poor_interface_modules:
            recommendations.append({
                'category': 'interface_improvement',
                'priority': 'medium',
                'modules': poor_interface_modules,
                'description': 'Improve interface design and documentation',
                'actions': [
                    'Add comprehensive docstrings to public methods',
                    'Simplify complex interfaces',
                    'Improve parameter consistency'
                ]
            })
        
        return recommendations
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for the analysis."""
        if not self.coupling_metrics:
            return {}
        
        instabilities = [m.instability for m in self.coupling_metrics.values()]
        distances = [m.distance for m in self.coupling_metrics.values()]
        cohesions = [m.cohesion_score for m in self.coupling_metrics.values()]
        interface_qualities = [m.interface_quality for m in self.coupling_metrics.values()]
        
        return {
            'instability': {
                'mean': sum(instabilities) / len(instabilities),
                'min': min(instabilities),
                'max': max(instabilities),
                'std': self._calculate_std(instabilities)
            },
            'distance': {
                'mean': sum(distances) / len(distances),
                'min': min(distances),
                'max': max(distances),
                'std': self._calculate_std(distances)
            },
            'cohesion': {
                'mean': sum(cohesions) / len(cohesions),
                'min': min(cohesions),
                'max': max(cohesions),
                'std': self._calculate_std(cohesions)
            },
            'interface_quality': {
                'mean': sum(interface_qualities) / len(interface_qualities),
                'min': min(interface_qualities),
                'max': max(interface_qualities),
                'std': self._calculate_std(interface_qualities)
            }
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def export_analysis(self, output_file: str = "coupling_analysis_results.json"):
        """Export coupling analysis results."""
        results = self.analyze_coupling_detailed()
        
        output_path = self.root_path / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Coupling analysis exported to {output_path}")
        return results


def main():
    """Main coupling analysis execution."""
    print("TestMaster Detailed Coupling Analysis - Agent B Phase 2 Hours 31-35")
    print("=" * 70)
    
    analyzer = DetailedCouplingAnalyzer()
    results = analyzer.export_analysis()
    
    # Print summary
    print("\nCOUPLING ANALYSIS SUMMARY:")
    print(f"Modules analyzed: {results['analysis_metadata']['modules_analyzed']}")
    print(f"Coupling issues found: {len(results['coupling_issues'])}")
    print(f"Recommendations generated: {len(results['recommendations'])}")
    
    # Print key metrics
    if results['summary_statistics']:
        stats = results['summary_statistics']
        print(f"\nKEY METRICS:")
        print(f"Average instability: {stats['instability']['mean']:.3f}")
        print(f"Average distance from main sequence: {stats['distance']['mean']:.3f}")
        print(f"Average cohesion: {stats['cohesion']['mean']:.3f}")
        print(f"Average interface quality: {stats['interface_quality']['mean']:.3f}")
    
    print("\nCoupling analysis complete!")
    return results


if __name__ == "__main__":
    main()