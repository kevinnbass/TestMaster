#!/usr/bin/env python3
"""
Base Generator Classes for TestMaster

Provides foundation classes for all test generators in the system.
Enhanced with shared state management and feature flags.
"""

import os
import sys
import ast
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import enhancement features
from ..core.feature_flags import FeatureFlags, is_shared_state_enabled
from ..core.shared_state import SharedState, get_shared_state

@dataclass
class ModuleAnalysis:
    """Analysis results for a module."""
    purpose: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    business_logic: str = ""
    edge_cases: List[str] = None
    dependencies: List[str] = None
    data_flows: str = ""
    error_scenarios: List[str] = None
    imports: List[str] = None
    has_main: bool = False
    
    def __post_init__(self):
        if self.edge_cases is None:
            self.edge_cases = []
        if self.dependencies is None:
            self.dependencies = []
        if self.error_scenarios is None:
            self.error_scenarios = []
        if self.imports is None:
            self.imports = []


@dataclass 
class GenerationConfig:
    """Configuration for test generation."""
    temperature: float = 0.1
    max_output_tokens: int = 4000
    use_real_imports: bool = True
    include_edge_cases: bool = True
    include_performance_tests: bool = False
    include_integration_tests: bool = True
    healing_iterations: int = 5
    quality_threshold: float = 70.0


class BaseGenerator(ABC):
    """Base class for all test generators."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize base generator with configuration."""
        self.config = config or GenerationConfig()
        self.stats = {
            "modules_processed": 0,
            "tests_generated": 0,
            "failures": 0,
            "start_time": time.time()
        }
        
        # NEW: Conditionally add shared state
        if is_shared_state_enabled():
            self.shared_state = get_shared_state()
            print("✅ SharedState enabled for test generation")
        else:
            self.shared_state = None
    
    @abstractmethod
    def analyze_module(self, module_path: Path) -> ModuleAnalysis:
        """Analyze a module to understand its structure and functionality."""
        pass
    
    @abstractmethod
    def generate_test_code(self, module_path: Path, analysis: ModuleAnalysis) -> str:
        """Generate test code based on module analysis."""
        pass
    
    def validate_test_code(self, test_code: str) -> bool:
        """Validate that generated test code has valid syntax."""
        try:
            ast.parse(test_code)
            return True
        except SyntaxError as e:
            print(f"Syntax error in generated test: {e}")
            return False
    
    def build_test_for_module(self, module_path: Path, output_dir: Path = None) -> bool:
        """Complete workflow to build test for a module."""
        print(f"\n{'='*60}")
        print(f"Building test for: {module_path.name}")
        print('='*60)
        
        # NEW: Check shared state for previous attempts
        if self.shared_state:
            module_key = str(module_path).replace('\\', '/')
            attempts = self.shared_state.get(f"attempts_{module_key}", 0)
            if attempts > 0:
                print(f"📊 Previous attempts: {attempts}")
            
            # Get previous test if exists
            previous_test = self.shared_state.get(f"last_test_{module_key}")
            if previous_test:
                print(f"💾 Found previous test in shared state")
        
        try:
            # Step 1: Analyze the module
            print("Step 1: Analyzing module...")
            analysis = self.analyze_module(module_path)
            
            if hasattr(analysis, 'error'):
                print(f"ERROR: Analysis failed: {analysis.error}")
                self.stats["failures"] += 1
                return False
            
            print(f"  Purpose: {analysis.purpose[:100]}...")
            print(f"  Classes: {len(analysis.classes)}")
            print(f"  Functions: {len(analysis.functions)}")
            
            # Step 2: Generate test code
            print("\nStep 2: Generating test code...")
            test_code = self.generate_test_code(module_path, analysis)
            
            # NEW: Update shared state with generated test
            if self.shared_state:
                module_key = str(module_path).replace('\\', '/')
                self.shared_state.set(f"last_test_{module_key}", test_code, ttl=3600)
                self.shared_state.increment(f"attempts_{module_key}")
            
            if not test_code or "Error generating test" in test_code:
                print("ERROR: Failed to generate test code")
                self.stats["failures"] += 1
                return False
            
            # Step 3: Validate test code
            print("\nStep 3: Validating test code...")
            if not self.validate_test_code(test_code):
                print("ERROR: Generated test has syntax errors")
                self.stats["failures"] += 1
                return False
            
            # Step 4: Save the test
            if output_dir is None:
                output_dir = Path("tests/unit")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            test_filename = f"test_{module_path.stem}.py"
            test_path = output_dir / test_filename
            
            test_path.write_text(test_code, encoding='utf-8')
            print(f"\nStep 4: Test saved to: {test_path}")
            
            # Update stats
            self.stats["modules_processed"] += 1
            self.stats["tests_generated"] += 1
            
            # NEW: Update shared state with success
            if self.shared_state:
                module_key = str(module_path).replace('\\', '/')
                self.shared_state.set(f"success_{module_key}", True)
                self.shared_state.set(f"test_path_{module_key}", str(test_path))
                
                # Track global stats in shared state
                self.shared_state.increment("global_tests_generated")
                self.shared_state.append("successful_modules", module_key)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Unexpected error in test generation: {e}")
            self.stats["failures"] += 1
            
            # NEW: Track failure in shared state
            if self.shared_state:
                module_key = str(module_path).replace('\\', '/')
                self.shared_state.set(f"last_error_{module_key}", str(e))
                self.shared_state.set(f"success_{module_key}", False)
                self.shared_state.increment("global_failures")
                self.shared_state.append("failed_modules", {
                    "module": module_key,
                    "error": str(e),
                    "timestamp": time.time()
                })
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        elapsed = time.time() - self.stats["start_time"]
        local_stats = {
            **self.stats,
            "elapsed_time": elapsed,
            "success_rate": (self.stats["modules_processed"] / 
                           max(1, self.stats["modules_processed"] + self.stats["failures"]) * 100)
        }
        
        # NEW: Include shared state statistics if enabled
        if self.shared_state:
            shared_stats = self.shared_state.get_stats()
            local_stats["shared_state"] = {
                "enabled": True,
                "backend": shared_stats.get("backend"),
                "total_keys": shared_stats.get("total_keys"),
                "hit_rate": shared_stats.get("hit_rate"),
                "global_tests": self.shared_state.get("global_tests_generated", 0),
                "global_failures": self.shared_state.get("global_failures", 0)
            }
        else:
            local_stats["shared_state"] = {"enabled": False}
        
        return local_stats
    
    def print_stats(self):
        """Print generation statistics."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print("GENERATION STATISTICS")
        print('='*60)
        print(f"Modules processed: {stats['modules_processed']}")
        print(f"Tests generated: {stats['tests_generated']}")
        print(f"Failures: {stats['failures']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Elapsed time: {stats['elapsed_time']:.1f}s")


class AnalysisBasedGenerator(BaseGenerator):
    """Base class for generators that perform detailed module analysis."""
    
    def analyze_module_ast(self, module_path: Path) -> ModuleAnalysis:
        """Perform AST-based analysis of module structure."""
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        content = module_path.read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in module: {e}")
        
        classes = []
        functions = []
        imports = []
        has_main = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            "name": item.name,
                            "args": [arg.arg for arg in item.args.args],
                            "is_private": item.name.startswith('_'),
                            "has_return": any(isinstance(n, ast.Return) for n in ast.walk(item))
                        })
                
                classes.append({
                    "name": node.name,
                    "methods": methods,
                    "bases": [self._unparse_node(base) for base in node.bases],
                    "line_number": node.lineno
                })
                
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                functions.append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "is_private": node.name.startswith('_'),
                    "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node)),
                    "line_number": node.lineno
                })
                
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Check for main block
        has_main = any(
            isinstance(node, ast.If) and 
            isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == "__name__"
            for node in ast.walk(tree)
        )
        
        return ModuleAnalysis(
            purpose=f"Module: {module_path.stem}",
            classes=classes,
            functions=functions,
            imports=imports,
            has_main=has_main,
            dependencies=list(set(imports))
        )
    
    def _unparse_node(self, node):
        """Safely unparse an AST node."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                # Fallback for older Python versions
                return str(node)
        except:
            return str(node)
    
    def build_import_path(self, module_path: Path) -> str:
        """Build proper import path for a module."""
        module_import = str(module_path).replace("\\", "/").replace(".py", "").replace("/", ".")
        
        # Handle common package structures
        if "testmaster" in module_import:
            idx = module_import.find("testmaster")
            module_import = module_import[idx:]
        elif "multi_coder_analysis" in module_import:
            idx = module_import.find("multi_coder_analysis")
            module_import = module_import[idx:]
        
        return module_import


class TemplateBasedGenerator(AnalysisBasedGenerator):
    """Base class for generators that use templates."""
    
    def generate_test_template(self, module_path: Path, analysis: ModuleAnalysis) -> str:
        """Generate a comprehensive test template."""
        module_name = module_path.stem
        module_import_path = self.build_import_path(module_path)
        
        template = f'''"""
Test suite for {module_name}

Auto-generated comprehensive test coverage.
Tests focus on real functionality without mocking internal components.
"""

import pytest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the module under test
try:
    import {module_import_path} as test_module
except ImportError as e:
    pytest.skip(f"Could not import {module_import_path}: {{e}}", allow_module_level=True)


class TestModuleStructure:
    """Test module structure and imports."""
    
    def test_module_imports(self):
        """Test that module imports successfully."""
        assert test_module is not None
        
    def test_module_attributes(self):
        """Test that module has expected attributes."""
'''
        
        # Add class tests
        for cls in analysis.classes:
            template += f'''        assert hasattr(test_module, "{cls['name']}"), "Missing class: {cls['name']}"\n'''
        
        # Add function tests
        for func in analysis.functions:
            if not func["is_private"]:
                template += f'''        assert hasattr(test_module, "{func['name']}"), "Missing function: {func['name']}"\n'''
        
        # Add class-specific test classes
        for cls in analysis.classes:
            template += f'''


class Test{cls['name']}:
    """Test {cls['name']} class."""
    
    def test_class_exists(self):
        """Test that {cls['name']} class exists."""
        assert hasattr(test_module, "{cls['name']}")
        cls_obj = getattr(test_module, "{cls['name']}")
        assert isinstance(cls_obj, type)
'''
            
            # Add method tests
            for method in cls.get("methods", []):
                if not method["is_private"] or method["name"] == "__init__":
                    safe_method_name = method['name'].replace('__', '_')
                    template += f'''
    def test_{safe_method_name}(self):
        """Test {cls['name']}.{method['name']} method."""
        cls_obj = getattr(test_module, "{cls['name']}")
        assert hasattr(cls_obj, "{method['name']}")
        # TODO: Add actual test implementation
'''
        
        # Add function tests
        if analysis.functions:
            template += '''


class TestModuleFunctions:
    """Test module-level functions."""
'''
            
            for func in analysis.functions:
                if not func["is_private"]:
                    template += f'''
    def test_{func['name']}(self):
        """Test {func['name']} function."""
        func_obj = getattr(test_module, "{func['name']}")
        assert callable(func_obj)
        # TODO: Add actual test implementation
'''
        
        template += '''


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''
        
        return template