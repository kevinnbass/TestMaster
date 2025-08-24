#!/usr/bin/env python3
"""
Component Analysis Script
========================

Comprehensive analysis of all components for consolidation planning.
This script ONLY analyzes and documents - it NEVER removes anything.

Part of Phase 1: Comprehensive Analysis & Mapping
"""

import ast
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import importlib.util
import inspect


@dataclass
class ComponentFunction:
    """Information about a function in a component."""
    name: str
    line_start: int
    line_end: int
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    complexity_score: int = 0


@dataclass
class ComponentClass:
    """Information about a class in a component."""
    name: str
    line_start: int
    line_end: int
    base_classes: List[str]
    methods: List[ComponentFunction]
    docstring: Optional[str]
    decorators: List[str] = field(default_factory=list)


@dataclass
class ComponentImport:
    """Information about an import in a component."""
    module: str
    names: List[str]
    alias: Optional[str]
    is_from_import: bool
    line_number: int


@dataclass
class ComponentAnalysis:
    """Complete analysis of a component."""
    file_path: str
    component_name: str
    analysis_timestamp: datetime
    
    # Code structure
    classes: List[ComponentClass]
    functions: List[ComponentFunction]
    imports: List[ComponentImport]
    
    # Dependencies
    external_dependencies: Set[str]
    internal_dependencies: Set[str]
    
    # Metrics
    lines_of_code: int
    complexity_score: int
    api_surface_area: int  # Number of public methods/functions
    
    # Content analysis
    docstring: Optional[str]
    has_tests: bool
    test_files: List[str] = field(default_factory=list)
    
    # Advanced features detected
    uses_ml: bool = False
    uses_async: bool = False
    uses_networking: bool = False
    uses_database: bool = False
    ml_libraries: Set[str] = field(default_factory=set)
    
    # Quality metrics
    documentation_coverage: float = 0.0
    estimated_migration_complexity: str = "unknown"


class ComponentAnalyzer:
    """
    Comprehensive component analyzer that inventories all functionality
    without removing or modifying anything.
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.analysis_results: Dict[str, ComponentAnalysis] = {}
        self.ml_keywords = {
            'sklearn', 'scipy', 'numpy', 'pandas', 'tensorflow', 'torch',
            'keras', 'xgboost', 'lightgbm', 'networkx', 'statsmodels'
        }
        self.async_keywords = {'async', 'await', 'asyncio', 'aiohttp'}
        self.network_keywords = {'requests', 'urllib', 'socket', 'http'}
        self.db_keywords = {'sql', 'database', 'sqlite', 'postgres', 'mongo'}
        
        print(f"[INFO] Component Analyzer initialized for: {base_path}")
    
    def analyze_all_components(self, target_paths: List[str]) -> Dict[str, ComponentAnalysis]:
        """Analyze all components in target paths."""
        print(f"[INFO] Starting comprehensive component analysis...")
        start_time = time.time()
        
        for target_path in target_paths:
            path = self.base_path / target_path
            if path.exists():
                print(f"[INFO] Analyzing components in: {target_path}")
                self._analyze_directory(path)
            else:
                print(f"[WARN] Path not found: {target_path}")
        
        duration = time.time() - start_time
        print(f"[INFO] Analysis complete: {len(self.analysis_results)} components analyzed in {duration:.2f}s")
        
        return self.analysis_results
    
    def _analyze_directory(self, directory: Path):
        """Recursively analyze all Python files in directory."""
        for py_file in directory.rglob("*.py"):
            if py_file.name.startswith('__'):
                continue
                
            try:
                analysis = self._analyze_file(py_file)
                if analysis:
                    self.analysis_results[str(py_file.relative_to(self.base_path))] = analysis
            except Exception as e:
                print(f"[ERROR] Failed to analyze {py_file}: {e}")
    
    def _analyze_file(self, file_path: Path) -> Optional[ComponentAnalysis]:
        """Analyze a single Python file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            analysis = ComponentAnalysis(
                file_path=str(file_path),
                component_name=file_path.stem,
                analysis_timestamp=datetime.now(),
                classes=[],
                functions=[],
                imports=[],
                external_dependencies=set(),
                internal_dependencies=set(),
                lines_of_code=len(content.splitlines()),
                complexity_score=0,
                api_surface_area=0,
                docstring=ast.get_docstring(tree),
                has_tests=False,
                test_files=[]
            )
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis.classes.append(self._analyze_class(node))
                elif isinstance(node, ast.FunctionDef):
                    analysis.functions.append(self._analyze_function(node))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis.imports.append(self._analyze_import(node))
            
            # Detect advanced features
            self._detect_advanced_features(content, analysis)
            
            # Calculate metrics
            self._calculate_metrics(analysis)
            
            # Find related test files
            self._find_test_files(file_path, analysis)
            
            return analysis
            
        except Exception as e:
            print(f"[ERROR] Error analyzing {file_path}: {e}")
            return None
    
    def _analyze_class(self, node: ast.ClassDef) -> ComponentClass:
        """Analyze a class definition."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._analyze_function(item))
        
        return ComponentClass(
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            base_classes=[base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
            methods=methods,
            docstring=ast.get_docstring(node),
            decorators=[d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
        )
    
    def _analyze_function(self, node: ast.FunctionDef) -> ComponentFunction:
        """Analyze a function definition."""
        args = [arg.arg for arg in node.args.args]
        
        # Calculate complexity (simplified)
        complexity = len(list(ast.walk(node)))
        
        return ComponentFunction(
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            args=args,
            returns=None,  # Would need more sophisticated analysis
            docstring=ast.get_docstring(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=[d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
            complexity_score=complexity
        )
    
    def _analyze_import(self, node) -> ComponentImport:
        """Analyze an import statement."""
        if isinstance(node, ast.Import):
            return ComponentImport(
                module=node.names[0].name,
                names=[alias.name for alias in node.names],
                alias=node.names[0].asname,
                is_from_import=False,
                line_number=node.lineno
            )
        else:  # ImportFrom
            return ComponentImport(
                module=node.module or "",
                names=[alias.name for alias in node.names],
                alias=None,
                is_from_import=True,
                line_number=node.lineno
            )
    
    def _detect_advanced_features(self, content: str, analysis: ComponentAnalysis):
        """Detect advanced features in the code."""
        content_lower = content.lower()
        
        # ML detection
        for ml_lib in self.ml_keywords:
            if ml_lib in content_lower:
                analysis.uses_ml = True
                analysis.ml_libraries.add(ml_lib)
        
        # Async detection
        for async_keyword in self.async_keywords:
            if async_keyword in content_lower:
                analysis.uses_async = True
                break
        
        # Network detection
        for net_keyword in self.network_keywords:
            if net_keyword in content_lower:
                analysis.uses_networking = True
                break
        
        # Database detection
        for db_keyword in self.db_keywords:
            if db_keyword in content_lower:
                analysis.uses_database = True
                break
        
        # Dependency classification
        for imp in analysis.imports:
            if any(keyword in imp.module.lower() for keyword in self.ml_keywords):
                analysis.external_dependencies.add(imp.module)
            elif imp.module.startswith('.') or 'testmaster' in imp.module.lower():
                analysis.internal_dependencies.add(imp.module)
            else:
                analysis.external_dependencies.add(imp.module)
    
    def _calculate_metrics(self, analysis: ComponentAnalysis):
        """Calculate various metrics for the component."""
        # API surface area (public methods/functions)
        analysis.api_surface_area = len([f for f in analysis.functions if not f.name.startswith('_')])
        for cls in analysis.classes:
            analysis.api_surface_area += len([m for m in cls.methods if not m.name.startswith('_')])
        
        # Complexity score
        analysis.complexity_score = sum(f.complexity_score for f in analysis.functions)
        for cls in analysis.classes:
            analysis.complexity_score += sum(m.complexity_score for m in cls.methods)
        
        # Documentation coverage
        documented_items = 0
        total_items = len(analysis.functions) + len(analysis.classes)
        
        if analysis.docstring:
            documented_items += 1
        documented_items += len([f for f in analysis.functions if f.docstring])
        documented_items += len([c for c in analysis.classes if c.docstring])
        
        if total_items > 0:
            analysis.documentation_coverage = documented_items / total_items
        
        # Migration complexity estimation
        if analysis.uses_ml and analysis.complexity_score > 500:
            analysis.estimated_migration_complexity = "high"
        elif analysis.uses_ml or analysis.complexity_score > 200:
            analysis.estimated_migration_complexity = "medium"
        else:
            analysis.estimated_migration_complexity = "low"
    
    def _find_test_files(self, file_path: Path, analysis: ComponentAnalysis):
        """Find related test files."""
        # Look for test files in common locations
        test_patterns = [
            f"test_{file_path.stem}.py",
            f"{file_path.stem}_test.py",
            f"test{file_path.stem}.py"
        ]
        
        # Search in tests/ directory and current directory
        search_dirs = [
            file_path.parent / "tests",
            file_path.parent.parent / "tests",
            file_path.parent,
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for pattern in test_patterns:
                    test_file = search_dir / pattern
                    if test_file.exists():
                        analysis.has_tests = True
                        analysis.test_files.append(str(test_file.relative_to(self.base_path)))
    
    def generate_report(self, output_file: str = "component_analysis_report.json"):
        """Generate comprehensive analysis report."""
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_components': len(self.analysis_results),
                'base_path': str(self.base_path)
            },
            'summary_statistics': self._generate_summary_stats(),
            'consolidation_priorities': self._identify_consolidation_priorities(),
            'components': {path: asdict(analysis) for path, analysis in self.analysis_results.items()}
        }
        
        # Write report
        output_path = self.base_path / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[INFO] Analysis report generated: {output_path}")
        return report
    
    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.analysis_results:
            return {}
        
        analyses = list(self.analysis_results.values())
        
        return {
            'total_lines_of_code': sum(a.lines_of_code for a in analyses),
            'total_classes': sum(len(a.classes) for a in analyses),
            'total_functions': sum(len(a.functions) for a in analyses),
            'components_with_ml': len([a for a in analyses if a.uses_ml]),
            'components_with_async': len([a for a in analyses if a.uses_async]),
            'components_with_tests': len([a for a in analyses if a.has_tests]),
            'average_complexity': sum(a.complexity_score for a in analyses) / len(analyses),
            'average_api_surface': sum(a.api_surface_area for a in analyses) / len(analyses),
            'migration_complexity_distribution': {
                'high': len([a for a in analyses if a.estimated_migration_complexity == 'high']),
                'medium': len([a for a in analyses if a.estimated_migration_complexity == 'medium']),
                'low': len([a for a in analyses if a.estimated_migration_complexity == 'low'])
            }
        }
    
    def _identify_consolidation_priorities(self) -> List[Dict[str, Any]]:
        """Identify consolidation priorities based on analysis."""
        priorities = []
        
        # Group by functionality
        analytics_components = [
            path for path, analysis in self.analysis_results.items()
            if 'analytics' in path.lower() or analysis.uses_ml
        ]
        
        testing_components = [
            path for path, analysis in self.analysis_results.items()
            if 'test' in path.lower() or 'coverage' in path.lower()
        ]
        
        integration_components = [
            path for path, analysis in self.analysis_results.items()
            if 'integration' in path.lower()
        ]
        
        if analytics_components:
            priorities.append({
                'category': 'analytics',
                'priority': 1,
                'components': analytics_components,
                'consolidation_target': 'core/intelligence/analytics/',
                'estimated_effort': 'high' if any(
                    self.analysis_results[comp].estimated_migration_complexity == 'high'
                    for comp in analytics_components
                ) else 'medium'
            })
        
        if testing_components:
            priorities.append({
                'category': 'testing',
                'priority': 1,
                'components': testing_components,
                'consolidation_target': 'core/intelligence/testing/',
                'estimated_effort': 'medium'
            })
        
        if integration_components:
            priorities.append({
                'category': 'integration',
                'priority': 2,
                'components': integration_components,
                'consolidation_target': 'core/intelligence/integration/',
                'estimated_effort': 'high'
            })
        
        return priorities


def main():
    """Main execution function."""
    print("=" * 80)
    print("COMPONENT ANALYSIS - PHASE 1: COMPREHENSIVE MAPPING")
    print("=" * 80)
    
    # Define target paths for analysis
    target_paths = [
        "integration",
        "testmaster/analysis", 
        "dashboard/dashboard_core",
        "core/testing",
        "core/analytics" if Path("core/analytics").exists() else None,
    ]
    
    # Remove None paths
    target_paths = [path for path in target_paths if path is not None]
    
    # Initialize analyzer
    base_path = Path(".")
    analyzer = ComponentAnalyzer(base_path)
    
    # Run analysis
    print(f"[INFO] Analyzing paths: {target_paths}")
    results = analyzer.analyze_all_components(target_paths)
    
    # Generate report
    report = analyzer.generate_report("phase1_component_analysis.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    stats = report['summary_statistics']
    print(f"Components analyzed: {report['analysis_metadata']['total_components']}")
    print(f"Total lines of code: {stats.get('total_lines_of_code', 0):,}")
    print(f"Total classes: {stats.get('total_classes', 0)}")
    print(f"Total functions: {stats.get('total_functions', 0)}")
    print(f"Components with ML: {stats.get('components_with_ml', 0)}")
    print(f"Components with tests: {stats.get('components_with_tests', 0)}")
    
    # Consolidation priorities
    priorities = report['consolidation_priorities']
    if priorities:
        print(f"\nCONSOLIDATION PRIORITIES:")
        for priority in priorities:
            print(f"  {priority['category'].upper()}: {len(priority['components'])} components")
            print(f"    Target: {priority['consolidation_target']}")
            print(f"    Effort: {priority['estimated_effort']}")
    
    print(f"\n[INFO] Detailed analysis saved to: phase1_component_analysis.json")
    print("[INFO] Phase 1 component analysis complete!")
    
    return results


if __name__ == '__main__':
    main()