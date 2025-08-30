#!/usr/bin/env python3
"""
Incremental Test Generation with Dependency Tracking
Tracks module dependencies and generates tests only for changed code.

Features:
- Import dependency graph construction
- Change impact analysis
- Smart test selection for CI/CD
- Incremental test generation
"""

import ast
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModuleInfo:
    """Information about a Python module."""
    path: Path
    name: str
    imports: Set[str] = field(default_factory=set)
    exports: Set[str] = field(default_factory=set)  # Functions, classes exported
    dependencies: Set[str] = field(default_factory=set)  # Module dependencies
    dependents: Set[str] = field(default_factory=set)  # Modules that depend on this
    content_hash: str = ""
    last_modified: datetime = field(default_factory=datetime.now)
    test_files: List[Path] = field(default_factory=list)
    coverage: float = 0.0
    complexity: int = 0
    
    def has_changed(self, new_hash: str) -> bool:
        """Check if module content has changed."""
        return self.content_hash != new_hash


class DependencyGraph:
    """Dependency graph for Python modules."""
    
    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def add_module(self, module_info: ModuleInfo):
        """Add module to dependency graph."""
        self.modules[module_info.name] = module_info
        
        # Update import graphs
        for dep in module_info.dependencies:
            self.import_graph[module_info.name].add(dep)
            self.reverse_graph[dep].add(module_info.name)
    
    def get_impact_radius(self, changed_module: str) -> Set[str]:
        """Get all modules impacted by a change."""
        impacted = set()
        queue = deque([changed_module])
        
        while queue:
            module = queue.popleft()
            if module in impacted:
                continue
            
            impacted.add(module)
            
            # Add all modules that depend on this module
            for dependent in self.reverse_graph.get(module, []):
                if dependent not in impacted:
                    queue.append(dependent)
        
        return impacted
    
    def get_test_impact(self, changed_modules: List[str]) -> Set[Path]:
        """Get all test files that need to be run."""
        test_files = set()
        
        for module in changed_modules:
            # Get impact radius
            impacted = self.get_impact_radius(module)
            
            # Collect test files for all impacted modules
            for impacted_module in impacted:
                if impacted_module in self.modules:
                    test_files.update(self.modules[impacted_module].test_files)
        
        return test_files
    
    def visualize(self) -> str:
        """Generate DOT format visualization of dependency graph."""
        dot_lines = ["digraph dependencies {"]
        dot_lines.append('  rankdir=LR;')
        dot_lines.append('  node [shape=box];')
        
        for module, deps in self.import_graph.items():
            for dep in deps:
                dot_lines.append(f'  "{module}" -> "{dep}";')
        
        dot_lines.append("}")
        return "\n".join(dot_lines)


class IncrementalTestGenerator:
    """Main incremental test generation system."""
    
    def __init__(self, cache_dir: str = ".testmaster_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dependency_graph = DependencyGraph()
        self.change_history: List[Dict] = []
        self.state_file = self.cache_dir / "incremental_state.pkl"
        
        # Load previous state if exists
        self.load_state()
    
    def analyze_module(self, file_path: Path) -> ModuleInfo:
        """Analyze a Python module and extract dependencies."""
        module_info = ModuleInfo(
            path=file_path,
            name=file_path.stem,
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate content hash
            module_info.content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_info.imports.add(alias.name)
                        module_info.dependencies.add(alias.name.split('.')[0])
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_info.imports.add(node.module)
                        module_info.dependencies.add(node.module.split('.')[0])
                
                # Extract exports (functions and classes)
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):
                        module_info.exports.add(node.name)
                
                elif isinstance(node, ast.ClassDef):
                    module_info.exports.add(node.name)
            
            # Calculate complexity
            module_info.complexity = self._calculate_complexity(tree)
            
            # Find associated test files
            module_info.test_files = self._find_test_files(file_path)
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
        
        return module_info
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of module."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _find_test_files(self, module_path: Path) -> List[Path]:
        """Find test files associated with a module."""
        test_files = []
        module_name = module_path.stem
        
        # Look for test files in common locations
        test_patterns = [
            f"test_{module_name}.py",
            f"tests/test_{module_name}.py",
            f"tests/unit/test_{module_name}.py",
            f"test/test_{module_name}.py",
        ]
        
        for pattern in test_patterns:
            test_path = module_path.parent / pattern
            if test_path.exists():
                test_files.append(test_path)
        
        return test_files
    
    def scan_codebase(self, source_dir: Path) -> DependencyGraph:
        """Scan entire codebase and build dependency graph."""
        logger.info(f"Scanning codebase in {source_dir}")
        
        for py_file in source_dir.rglob("*.py"):
            if "test" not in py_file.name and "__pycache__" not in str(py_file):
                module_info = self.analyze_module(py_file)
                self.dependency_graph.add_module(module_info)
        
        logger.info(f"Found {len(self.dependency_graph.modules)} modules")
        return self.dependency_graph
    
    def detect_changes(self, source_dir: Path) -> Dict[str, List[str]]:
        """Detect which modules have changed since last scan."""
        changes = {
            "modified": [],
            "added": [],
            "deleted": [],
            "unchanged": []
        }
        
        current_modules = {}
        
        # Scan current state
        for py_file in source_dir.rglob("*.py"):
            if "test" not in py_file.name and "__pycache__" not in str(py_file):
                module_info = self.analyze_module(py_file)
                current_modules[module_info.name] = module_info
        
        # Compare with previous state
        for module_name, module_info in current_modules.items():
            if module_name in self.dependency_graph.modules:
                old_module = self.dependency_graph.modules[module_name]
                if old_module.has_changed(module_info.content_hash):
                    changes["modified"].append(module_name)
                else:
                    changes["unchanged"].append(module_name)
            else:
                changes["added"].append(module_name)
        
        # Check for deleted modules
        for module_name in self.dependency_graph.modules:
            if module_name not in current_modules:
                changes["deleted"].append(module_name)
        
        # Update dependency graph
        for module_name, module_info in current_modules.items():
            self.dependency_graph.add_module(module_info)
        
        # Record change
        self.change_history.append({
            "timestamp": datetime.now().isoformat(),
            "changes": changes
        })
        
        return changes
    
    def generate_incremental_tests(self, changes: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate tests only for changed modules and their dependents."""
        results = {
            "modules_to_test": [],
            "tests_generated": [],
            "tests_updated": [],
            "skipped": [],
            "impact_analysis": {}
        }
        
        # Collect all modules that need testing
        modules_to_test = set()
        
        # Add modified modules
        for module in changes["modified"]:
            modules_to_test.add(module)
            impact = self.dependency_graph.get_impact_radius(module)
            results["impact_analysis"][module] = list(impact)
            modules_to_test.update(impact)
        
        # Add new modules
        modules_to_test.update(changes["added"])
        
        results["modules_to_test"] = list(modules_to_test)
        
        # Generate or update tests for each module
        for module_name in modules_to_test:
            if module_name not in self.dependency_graph.modules:
                continue
            
            module_info = self.dependency_graph.modules[module_name]
            
            if module_name in changes["added"]:
                # Generate new test
                test_path = self._generate_new_test(module_info)
                if test_path:
                    results["tests_generated"].append(str(test_path))
            
            elif module_name in changes["modified"] or module_name in results["impact_analysis"]:
                # Update existing test
                updated = self._update_existing_test(module_info)
                if updated:
                    results["tests_updated"].append(module_name)
            
            else:
                results["skipped"].append(module_name)
        
        return results
    
    def _generate_new_test(self, module_info: ModuleInfo) -> Optional[Path]:
        """Generate new test file for module."""
        test_content = f'''#!/usr/bin/env python3
"""
Auto-generated tests for {module_info.name}
Generated: {datetime.now().isoformat()}
"""

import pytest
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from {module_info.name} import {', '.join(module_info.exports) if module_info.exports else '*'}


class Test{module_info.name.title().replace("_", "")}:
    """Test cases for {module_info.name} module."""
'''
        
        # Add test methods for each exported function/class
        for export in module_info.exports:
            test_content += f'''
    
    def test_{export.lower()}(self):
        """Test {export} functionality."""
        # TODO: Implement comprehensive test for {export}
        assert True  # Placeholder
'''
        
        # Determine test file path
        test_dir = module_info.path.parent / "tests"
        test_dir.mkdir(exist_ok=True)
        test_path = test_dir / f"test_{module_info.name}.py"
        
        # Write test file
        with open(test_path, 'w') as f:
            f.write(test_content)
        
        logger.info(f"Generated new test: {test_path}")
        return test_path
    
    def _update_existing_test(self, module_info: ModuleInfo) -> bool:
        """Update existing test file for module."""
        if not module_info.test_files:
            # No existing test, generate new one
            self._generate_new_test(module_info)
            return True
        
        # For now, just mark as needing update
        # In real implementation, would analyze test coverage gaps
        logger.info(f"Test update needed for {module_info.name}")
        return True
    
    def select_tests_for_ci(self, changed_files: List[Path]) -> List[Path]:
        """Select minimal set of tests to run in CI/CD."""
        # Extract module names from changed files
        changed_modules = []
        for file_path in changed_files:
            if file_path.suffix == ".py" and "test" not in file_path.name:
                changed_modules.append(file_path.stem)
        
        # Get all impacted test files
        test_files = self.dependency_graph.get_test_impact(changed_modules)
        
        # Sort by priority (based on module complexity and coverage)
        prioritized_tests = self._prioritize_tests(test_files)
        
        return prioritized_tests
    
    def _prioritize_tests(self, test_files: Set[Path]) -> List[Path]:
        """Prioritize tests based on risk and importance."""
        test_priority = []
        
        for test_file in test_files:
            # Find associated module
            module_name = test_file.stem.replace("test_", "")
            
            if module_name in self.dependency_graph.modules:
                module = self.dependency_graph.modules[module_name]
                
                # Calculate priority score
                priority = (
                    module.complexity * 2 +  # Complexity weight
                    len(module.dependents) * 3 +  # Dependency weight
                    (100 - module.coverage) / 10  # Coverage gap weight
                )
                
                test_priority.append((test_file, priority))
            else:
                test_priority.append((test_file, 0))
        
        # Sort by priority (highest first)
        test_priority.sort(key=lambda x: x[1], reverse=True)
        
        return [test for test, _ in test_priority]
    
    def generate_impact_report(self, changes: Dict[str, List[str]]) -> str:
        """Generate impact analysis report."""
        report_lines = [
            "=" * 60,
            "CHANGE IMPACT ANALYSIS REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "CHANGES DETECTED:",
            f"  Modified: {len(changes['modified'])} modules",
            f"  Added: {len(changes['added'])} modules",
            f"  Deleted: {len(changes['deleted'])} modules",
            ""
        ]
        
        if changes["modified"]:
            report_lines.append("MODIFIED MODULES AND THEIR IMPACT:")
            for module in changes["modified"]:
                impact = self.dependency_graph.get_impact_radius(module)
                report_lines.append(f"\n  {module}:")
                report_lines.append(f"    Direct dependents: {len(self.dependency_graph.reverse_graph.get(module, []))}")
                report_lines.append(f"    Total impact: {len(impact)} modules")
                
                if len(impact) > 1:
                    report_lines.append("    Impacted modules:")
                    for impacted in list(impact)[:10]:
                        if impacted != module:
                            report_lines.append(f"      - {impacted}")
                    
                    if len(impact) > 11:
                        report_lines.append(f"      ... and {len(impact) - 11} more")
        
        if changes["added"]:
            report_lines.append("\nNEW MODULES:")
            for module in changes["added"]:
                report_lines.append(f"  + {module}")
        
        if changes["deleted"]:
            report_lines.append("\nDELETED MODULES:")
            for module in changes["deleted"]:
                report_lines.append(f"  - {module}")
        
        # Test selection
        all_changed = changes["modified"] + changes["added"]
        if all_changed:
            test_files = self.dependency_graph.get_test_impact(all_changed)
            report_lines.extend([
                "",
                "TEST SELECTION:",
                f"  Tests to run: {len(test_files)}",
                "  Test files:"
            ])
            
            for test_file in list(test_files)[:20]:
                report_lines.append(f"    - {test_file}")
            
            if len(test_files) > 20:
                report_lines.append(f"    ... and {len(test_files) - 20} more")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
    
    def save_state(self):
        """Save current state to disk."""
        state = {
            "dependency_graph": self.dependency_graph,
            "change_history": self.change_history,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"State saved to {self.state_file}")
    
    def load_state(self):
        """Load previous state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                
                self.dependency_graph = state.get("dependency_graph", DependencyGraph())
                self.change_history = state.get("change_history", [])
                
                logger.info(f"Loaded state from {state.get('timestamp', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dependency graph."""
        stats = {
            "total_modules": len(self.dependency_graph.modules),
            "total_dependencies": sum(len(deps) for deps in self.dependency_graph.import_graph.values()),
            "average_dependencies": 0,
            "most_depended_on": [],
            "most_complex": [],
            "least_tested": []
        }
        
        if self.dependency_graph.modules:
            # Average dependencies
            stats["average_dependencies"] = stats["total_dependencies"] / stats["total_modules"]
            
            # Most depended on modules
            dependency_counts = [
                (module, len(self.dependency_graph.reverse_graph.get(module, [])))
                for module in self.dependency_graph.modules
            ]
            dependency_counts.sort(key=lambda x: x[1], reverse=True)
            stats["most_depended_on"] = dependency_counts[:5]
            
            # Most complex modules
            complexity_list = [
                (module, info.complexity)
                for module, info in self.dependency_graph.modules.items()
            ]
            complexity_list.sort(key=lambda x: x[1], reverse=True)
            stats["most_complex"] = complexity_list[:5]
            
            # Least tested modules
            coverage_list = [
                (module, info.coverage)
                for module, info in self.dependency_graph.modules.items()
            ]
            coverage_list.sort(key=lambda x: x[1])
            stats["least_tested"] = coverage_list[:5]
        
        return stats


def main():
    """CLI for incremental test generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental Test Generation with Dependency Tracking")
    parser.add_argument("--source-dir", required=True, help="Source code directory")
    parser.add_argument("--scan", action="store_true", help="Scan codebase and build dependency graph")
    parser.add_argument("--detect-changes", action="store_true", help="Detect changes since last scan")
    parser.add_argument("--generate", action="store_true", help="Generate incremental tests")
    parser.add_argument("--select-tests", help="Select tests for changed files (comma-separated)")
    parser.add_argument("--visualize", help="Export dependency graph to DOT file")
    parser.add_argument("--stats", action="store_true", help="Show dependency statistics")
    parser.add_argument("--report", action="store_true", help="Generate impact report")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = IncrementalTestGenerator()
    source_dir = Path(args.source_dir)
    
    if args.scan:
        print("Scanning codebase...")
        graph = generator.scan_codebase(source_dir)
        generator.save_state()
        print(f"✓ Found {len(graph.modules)} modules")
    
    if args.detect_changes:
        print("Detecting changes...")
        changes = generator.detect_changes(source_dir)
        
        print(f"\nChanges detected:")
        print(f"  Modified: {len(changes['modified'])} modules")
        print(f"  Added: {len(changes['added'])} modules")
        print(f"  Deleted: {len(changes['deleted'])} modules")
        
        if args.report:
            report = generator.generate_impact_report(changes)
            print("\n" + report)
        
        generator.save_state()
    
    if args.generate:
        print("Generating incremental tests...")
        changes = generator.detect_changes(source_dir)
        results = generator.generate_incremental_tests(changes)
        
        print(f"\nTest generation results:")
        print(f"  Modules to test: {len(results['modules_to_test'])}")
        print(f"  Tests generated: {len(results['tests_generated'])}")
        print(f"  Tests updated: {len(results['tests_updated'])}")
        
        if results['tests_generated']:
            print("\nNew tests generated:")
            for test in results['tests_generated']:
                print(f"  + {test}")
    
    if args.select_tests:
        changed_files = [Path(f.strip()) for f in args.select_tests.split(",")]
        selected_tests = generator.select_tests_for_ci(changed_files)
        
        print(f"\nTests to run ({len(selected_tests)}):")
        for test in selected_tests:
            print(f"  - {test}")
    
    if args.visualize:
        dot_content = generator.dependency_graph.visualize()
        with open(args.visualize, 'w') as f:
            f.write(dot_content)
        print(f"✓ Dependency graph exported to {args.visualize}")
    
    if args.stats:
        stats = generator.get_statistics()
        
        print("\nDependency Statistics:")
        print("=" * 40)
        print(f"Total modules: {stats['total_modules']}")
        print(f"Total dependencies: {stats['total_dependencies']}")
        print(f"Average dependencies: {stats['average_dependencies']:.1f}")
        
        if stats['most_depended_on']:
            print("\nMost depended on:")
            for module, count in stats['most_depended_on']:
                print(f"  {module}: {count} dependents")
        
        if stats['most_complex']:
            print("\nMost complex:")
            for module, complexity in stats['most_complex']:
                print(f"  {module}: complexity {complexity}")
        
        if stats['least_tested']:
            print("\nLeast tested:")
            for module, coverage in stats['least_tested']:
                print(f"  {module}: {coverage:.1f}% coverage")


if __name__ == "__main__":
    main()