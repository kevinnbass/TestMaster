"""
Test-Module Mapping System

Inspired by Agency-Swarm's hierarchical thread mapping pattern:
- Bidirectional relationships like agent_name -> {other_agent: {thread_data}}
- Real-time updates and cache management
- Support for complex many-to-many relationships

Maps:
- test_file -> {module_1: {coverage_data}, module_2: {coverage_data}}
- module_file -> {test_1: {relationship_data}, test_2: {relationship_data}}
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import re

from ..core.layer_manager import requires_layer


@dataclass
class ModuleCoverage:
    """Coverage information for a module within a test."""
    module_path: str
    lines_covered: Set[int]
    total_lines: Set[int] 
    coverage_percentage: float
    functions_covered: Set[str]
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TestRelationship:
    """Relationship data between a test and module."""
    test_path: str
    relationship_type: str  # "unit", "integration", "functional", "e2e"
    import_patterns: List[str]
    functions_tested: Set[str]
    last_run: Optional[datetime] = None
    last_passed: Optional[bool] = None
    priority: int = 1  # 1=low, 5=critical
    

@dataclass
class TestModuleMapping:
    """Complete bidirectional mapping between tests and modules."""
    # Test -> Module mappings (like Agency-Swarm's agent relationships)
    test_to_modules: Dict[str, Dict[str, ModuleCoverage]] = field(default_factory=lambda: defaultdict(dict))
    
    # Module -> Test mappings (reverse lookup)
    module_to_tests: Dict[str, Dict[str, TestRelationship]] = field(default_factory=lambda: defaultdict(dict))
    
    # Integration tests -> Multiple modules
    integration_mappings: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    total_tests: int = 0
    total_modules: int = 0
    

class TestMapper:
    """
    Bidirectional test-module mapper using Agency-Swarm hierarchical patterns.
    
    Maintains real-time mappings between tests and source modules,
    similar to how Agency-Swarm manages agent-to-agent relationships.
    """
    
    @requires_layer("layer1_test_foundation", "test_mapping")
    def __init__(self, source_dir: Union[str, Path], test_dir: Union[str, Path]):
        """
        Initialize the test mapper.
        
        Args:
            source_dir: Directory containing source code
            test_dir: Directory containing tests
        """
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.cache_file = Path("testmaster_mapping_cache.json")
        
        # Initialize mapping structure (Agency-Swarm pattern)
        self.mapping = TestModuleMapping()
        
        # Load existing mapping from cache
        self._load_mapping_cache()
        
        # Analysis caches
        self._ast_cache = {}
        self._import_cache = {}
    
    def build_complete_mapping(self) -> TestModuleMapping:
        """
        Build complete bidirectional mapping between tests and modules.
        
        Returns:
            Complete mapping with both directions populated
        """
        print("🗺️ Building complete test-module mapping...")
        
        # Clear existing mappings
        self.mapping = TestModuleMapping()
        
        # Step 1: Discover all source modules
        modules = self._discover_source_modules()
        print(f"📁 Found {len(modules)} source modules")
        
        # Step 2: Discover all test files
        tests = self._discover_test_files()
        print(f"🧪 Found {len(tests)} test files")
        
        # Step 3: Build test -> module mappings
        self._build_test_to_module_mappings(tests, modules)
        
        # Step 4: Build module -> test mappings (reverse)
        self._build_module_to_test_mappings()
        
        # Step 5: Identify integration tests
        self._identify_integration_tests()
        
        # Update metadata
        self.mapping.total_tests = len(tests)
        self.mapping.total_modules = len(modules)
        self.mapping.last_updated = datetime.now()
        
        # Save to cache
        self._save_mapping_cache()
        
        self._print_mapping_summary()
        return self.mapping
    
    def _discover_source_modules(self) -> List[Path]:
        """Discover all source modules to map."""
        modules = []
        
        for py_file in self.source_dir.rglob("*.py"):
            if self._should_map_module(py_file):
                modules.append(py_file)
        
        return sorted(modules)
    
    def _should_map_module(self, file_path: Path) -> bool:
        """Check if module should be included in mapping."""
        if "__pycache__" in str(file_path):
            return False
        if file_path.name.startswith("_") and file_path.stem != "__init__":
            return False
        if "test" in file_path.name.lower():
            return False
        return True
    
    def _discover_test_files(self) -> List[Path]:
        """Discover all test files to map."""
        tests = []
        
        for py_file in self.test_dir.rglob("*.py"):
            if self._should_map_test(py_file):
                tests.append(py_file)
        
        return sorted(tests)
    
    def _should_map_test(self, file_path: Path) -> bool:
        """Check if test file should be included in mapping."""
        if "__pycache__" in str(file_path):
            return False
        # Most test files start with 'test_' or end with '_test.py'
        name = file_path.name.lower()
        return name.startswith("test_") or name.endswith("_test.py")
    
    def _build_test_to_module_mappings(self, tests: List[Path], modules: List[Path]):
        """Build test -> module mappings by analyzing imports and content."""
        print("🔍 Analyzing test -> module relationships...")
        
        for test_file in tests:
            try:
                mapped_modules = self._analyze_test_module_relationships(test_file, modules)
                
                test_key = str(test_file.relative_to(self.test_dir))
                
                for module_file, coverage_info in mapped_modules.items():
                    module_key = str(module_file.relative_to(self.source_dir))
                    self.mapping.test_to_modules[test_key][module_key] = coverage_info
                    
            except Exception as e:
                print(f"⚠️ Error mapping test {test_file}: {e}")
    
    def _analyze_test_module_relationships(self, test_file: Path, 
                                          modules: List[Path]) -> Dict[Path, ModuleCoverage]:
        """Analyze which modules a test file relates to."""
        relationships = {}
        
        try:
            # Parse test file AST
            with open(test_file, 'r', encoding='utf-8') as f:
                test_content = f.read()
                test_tree = ast.parse(test_content)
            
            # Find import statements
            imports = self._extract_imports(test_tree)
            
            # Find function calls and references
            references = self._extract_code_references(test_tree)
            
            # Match imports and references to modules
            for module_file in modules:
                coverage_info = self._calculate_module_coverage(
                    module_file, test_file, imports, references, test_content
                )
                
                if coverage_info and coverage_info.coverage_percentage > 0:
                    relationships[module_file] = coverage_info
                    
        except Exception as e:
            print(f"⚠️ Error analyzing {test_file}: {e}")
        
        return relationships
    
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
    
    def _extract_code_references(self, tree: ast.AST) -> Set[str]:
        """Extract function calls and variable references."""
        references = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    references.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    references.add(node.func.attr)
            elif isinstance(node, ast.Name):
                references.add(node.id)
        
        return references
    
    def _calculate_module_coverage(self, module_file: Path, test_file: Path,
                                  imports: List[str], references: Set[str],
                                  test_content: str) -> Optional[ModuleCoverage]:
        """Calculate how much of a module is covered by a test."""
        try:
            module_name = module_file.stem
            module_rel_path = str(module_file.relative_to(self.source_dir))
            
            # Check if module is directly imported
            direct_import = any(
                module_name in imp or module_rel_path.replace('/', '.').replace('.py', '') in imp
                for imp in imports
            )
            
            # Check if module functions are referenced
            with open(module_file, 'r', encoding='utf-8') as f:
                module_tree = ast.parse(f.read())
            
            module_functions = {
                node.name for node in ast.walk(module_tree) 
                if isinstance(node, ast.FunctionDef)
            }
            
            referenced_functions = module_functions & references
            
            # Check for string-based references (test file content mentions)
            content_mentions = test_content.count(module_name)
            
            # Calculate coverage score
            coverage_score = 0
            if direct_import:
                coverage_score += 50
            if referenced_functions:
                coverage_score += 30 * (len(referenced_functions) / max(len(module_functions), 1))
            if content_mentions > 0:
                coverage_score += min(20, content_mentions * 5)
            
            if coverage_score < 10:  # Minimum threshold
                return None
            
            # Get all executable lines in module (simplified)
            module_lines = self._get_executable_lines(module_tree)
            
            return ModuleCoverage(
                module_path=str(module_file),
                lines_covered=set(),  # Would need actual execution to fill
                total_lines=module_lines,
                coverage_percentage=min(coverage_score, 100),
                functions_covered=referenced_functions
            )
            
        except Exception as e:
            print(f"⚠️ Error calculating coverage for {module_file}: {e}")
            return None
    
    def _get_executable_lines(self, tree: ast.AST) -> Set[int]:
        """Get line numbers that contain executable code."""
        lines = set()
        
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                lines.add(node.lineno)
        
        return lines
    
    def _build_module_to_test_mappings(self):
        """Build module -> test mappings (reverse lookup)."""
        print("🔄 Building reverse module -> test mappings...")
        
        # Clear existing reverse mappings
        self.mapping.module_to_tests = defaultdict(dict)
        
        # Build reverse mapping from test -> module data
        for test_path, module_dict in self.mapping.test_to_modules.items():
            for module_path, coverage_info in module_dict.items():
                
                # Determine relationship type
                relationship_type = self._determine_relationship_type(
                    test_path, module_path, coverage_info
                )
                
                # Create relationship object
                relationship = TestRelationship(
                    test_path=test_path,
                    relationship_type=relationship_type,
                    import_patterns=[],  # Could extract from coverage_info
                    functions_tested=coverage_info.functions_covered,
                    priority=self._calculate_test_priority(coverage_info)
                )
                
                self.mapping.module_to_tests[module_path][test_path] = relationship
    
    def _determine_relationship_type(self, test_path: str, module_path: str, 
                                   coverage_info: ModuleCoverage) -> str:
        """Determine the type of relationship between test and module."""
        test_name = Path(test_path).stem.lower()
        module_name = Path(module_path).stem.lower()
        
        # Integration tests
        if "integration" in test_path.lower() or "e2e" in test_path.lower():
            return "integration"
        
        # Direct unit tests (test_module.py -> module.py)
        if f"test_{module_name}" == test_name or f"{module_name}_test" == test_name:
            return "unit"
        
        # Functional tests
        if "functional" in test_path.lower() or "feature" in test_path.lower():
            return "functional"
        
        # Default to unit if high coverage, integration if low
        return "unit" if coverage_info.coverage_percentage > 70 else "integration"
    
    def _calculate_test_priority(self, coverage_info: ModuleCoverage) -> int:
        """Calculate priority level for a test (1=low, 5=critical)."""
        if coverage_info.coverage_percentage > 90:
            return 5  # Critical - high coverage
        elif coverage_info.coverage_percentage > 70:
            return 4  # High
        elif coverage_info.coverage_percentage > 50:
            return 3  # Medium
        elif coverage_info.coverage_percentage > 25:
            return 2  # Low
        else:
            return 1  # Very low
    
    def _identify_integration_tests(self):
        """Identify tests that cover multiple modules (integration tests)."""
        print("🔗 Identifying integration tests...")
        
        for test_path, module_dict in self.mapping.test_to_modules.items():
            if len(module_dict) > 1:  # Test covers multiple modules
                modules = list(module_dict.keys())
                self.mapping.integration_mappings[test_path] = modules
                
                print(f"  📊 {test_path} -> {len(modules)} modules")
    
    def _load_mapping_cache(self):
        """Load mapping from cache file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct mapping from JSON data
                self.mapping = self._deserialize_mapping(data)
                print(f"📥 Loaded mapping cache from {self.cache_file}")
                
            except Exception as e:
                print(f"⚠️ Error loading mapping cache: {e}")
    
    def _save_mapping_cache(self):
        """Save mapping to cache file."""
        try:
            data = self._serialize_mapping(self.mapping)
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"💾 Saved mapping cache to {self.cache_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving mapping cache: {e}")
    
    def _serialize_mapping(self, mapping: TestModuleMapping) -> Dict[str, Any]:
        """Convert mapping to JSON-serializable format."""
        return {
            "test_to_modules": {
                test: {
                    module: {
                        "module_path": cov.module_path,
                        "lines_covered": list(cov.lines_covered),
                        "total_lines": list(cov.total_lines),
                        "coverage_percentage": cov.coverage_percentage,
                        "functions_covered": list(cov.functions_covered),
                        "last_updated": cov.last_updated.isoformat()
                    }
                    for module, cov in modules.items()
                }
                for test, modules in mapping.test_to_modules.items()
            },
            "module_to_tests": {
                module: {
                    test: {
                        "test_path": rel.test_path,
                        "relationship_type": rel.relationship_type,
                        "import_patterns": rel.import_patterns,
                        "functions_tested": list(rel.functions_tested),
                        "priority": rel.priority,
                        "last_run": rel.last_run.isoformat() if rel.last_run else None,
                        "last_passed": rel.last_passed
                    }
                    for test, rel in tests.items()
                }
                for module, tests in mapping.module_to_tests.items()
            },
            "integration_mappings": mapping.integration_mappings,
            "last_updated": mapping.last_updated.isoformat(),
            "total_tests": mapping.total_tests,
            "total_modules": mapping.total_modules
        }
    
    def _deserialize_mapping(self, data: Dict[str, Any]) -> TestModuleMapping:
        """Convert JSON data back to mapping objects."""
        mapping = TestModuleMapping()
        
        # Deserialize test_to_modules
        for test, modules in data.get("test_to_modules", {}).items():
            for module, cov_data in modules.items():
                coverage = ModuleCoverage(
                    module_path=cov_data["module_path"],
                    lines_covered=set(cov_data["lines_covered"]),
                    total_lines=set(cov_data["total_lines"]),
                    coverage_percentage=cov_data["coverage_percentage"],
                    functions_covered=set(cov_data["functions_covered"]),
                    last_updated=datetime.fromisoformat(cov_data["last_updated"])
                )
                mapping.test_to_modules[test][module] = coverage
        
        # Deserialize module_to_tests
        for module, tests in data.get("module_to_tests", {}).items():
            for test, rel_data in tests.items():
                relationship = TestRelationship(
                    test_path=rel_data["test_path"],
                    relationship_type=rel_data["relationship_type"],
                    import_patterns=rel_data["import_patterns"],
                    functions_tested=set(rel_data["functions_tested"]),
                    priority=rel_data["priority"],
                    last_run=datetime.fromisoformat(rel_data["last_run"]) if rel_data["last_run"] else None,
                    last_passed=rel_data["last_passed"]
                )
                mapping.module_to_tests[module][test] = relationship
        
        # Copy other fields
        mapping.integration_mappings = data.get("integration_mappings", {})
        mapping.last_updated = datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        mapping.total_tests = data.get("total_tests", 0)
        mapping.total_modules = data.get("total_modules", 0)
        
        return mapping
    
    def _print_mapping_summary(self):
        """Print summary of mapping results."""
        print("\n" + "="*60)
        print("🗺️ TEST-MODULE MAPPING SUMMARY")
        print("="*60)
        
        print(f"📊 Total Modules: {self.mapping.total_modules}")
        print(f"🧪 Total Tests: {self.mapping.total_tests}")
        print(f"🔗 Integration Tests: {len(self.mapping.integration_mappings)}")
        
        # Show coverage distribution
        covered_modules = len(self.mapping.module_to_tests)
        uncovered_modules = self.mapping.total_modules - covered_modules
        
        print(f"\n📈 Coverage Distribution:")
        print(f"   Modules with tests: {covered_modules}")
        print(f"   Modules without tests: {uncovered_modules}")
        
        if covered_modules > 0:
            coverage_rate = (covered_modules / self.mapping.total_modules) * 100
            print(f"   Overall coverage rate: {coverage_rate:.1f}%")
        
        # Show relationship types
        relationship_counts = defaultdict(int)
        for tests in self.mapping.module_to_tests.values():
            for relationship in tests.values():
                relationship_counts[relationship.relationship_type] += 1
        
        if relationship_counts:
            print(f"\n🔍 Relationship Types:")
            for rel_type, count in relationship_counts.items():
                print(f"   {rel_type.title()}: {count}")
        
        print("="*60)
    
    # Convenience methods for querying mappings
    
    def get_tests_for_module(self, module_path: str) -> List[str]:
        """Get all tests that cover a specific module."""
        return list(self.mapping.module_to_tests.get(module_path, {}).keys())
    
    def get_modules_for_test(self, test_path: str) -> List[str]:
        """Get all modules covered by a specific test."""
        return list(self.mapping.test_to_modules.get(test_path, {}).keys())
    
    def get_uncovered_modules(self) -> List[str]:
        """Get modules that have no test coverage."""
        all_modules = set()
        # Would need to scan filesystem to get complete list
        covered_modules = set(self.mapping.module_to_tests.keys())
        return list(all_modules - covered_modules)
    
    def get_integration_tests(self) -> Dict[str, List[str]]:
        """Get all integration tests and their covered modules."""
        return dict(self.mapping.integration_mappings)
    
    def update_test_result(self, test_path: str, passed: bool):
        """Update test result for all related modules."""
        if test_path in self.mapping.test_to_modules:
            for module_path in self.mapping.test_to_modules[test_path]:
                if module_path in self.mapping.module_to_tests:
                    if test_path in self.mapping.module_to_tests[module_path]:
                        relationship = self.mapping.module_to_tests[module_path][test_path]
                        relationship.last_run = datetime.now()
                        relationship.last_passed = passed