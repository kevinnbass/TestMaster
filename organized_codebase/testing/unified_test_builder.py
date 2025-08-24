#!/usr/bin/env python3
"""
Unified Intelligent Test Builder
Consolidates functionality from all test builder variants into a single, cohesive framework.

This unified builder combines:
- AI-powered test generation (from intelligent_test_builder.py)
- Offline/template generation (from intelligent_test_builder_offline.py)
- Direct SDK integration (from intelligent_test_builder_v2.py)
"""

import os
import sys
import ast
import json
import time
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import pickle
import hashlib
from collections import defaultdict, deque
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class TestBuilderMode(Enum):
    """Modes of operation for the test builder."""
    AI_POWERED = "ai_powered"      # Use Gemini API for intelligent generation
    OFFLINE = "offline"             # Generate templates without API
    HYBRID = "hybrid"               # Combine offline analysis with AI enhancement


class TestStrategy(Enum):
    """Test generation strategies."""
    COMPREHENSIVE = "comprehensive"  # Full test coverage with edge cases
    QUICK = "quick"                 # Basic functionality tests
    FOCUSED = "focused"             # Target specific functions/classes
    REGRESSION = "regression"       # Tests for bug fixes and changes


class MetaLearningEngine:
    """
    Meta-Learning Engine for Intelligent Test Generation
    Agent C Phase 2 Enhancement: Hours 100-110 Meta-Learning System Development
    
    Implements learning-to-learn algorithms for adaptive test generation.
    """
    
    def __init__(self, memory_size: int = 10000):
        """Initialize meta-learning engine with experience memory."""
        self.memory_size = memory_size
        self.experience_memory = deque(maxlen=memory_size)
        self.pattern_cache = {}
        self.adaptation_history = defaultdict(list)
        self.performance_metrics = defaultdict(float)
        self.learning_rate = 0.1
        
    def record_experience(self, module_info: Dict[str, Any], 
                         test_strategy: TestStrategy, 
                         test_quality: float, 
                         execution_time: float):
        """Record test generation experience for learning."""
        experience = {
            'timestamp': time.time(),
            'module_hash': self._hash_module_info(module_info),
            'module_features': self._extract_features(module_info),
            'strategy': test_strategy,
            'quality_score': test_quality,
            'execution_time': execution_time,
            'success': test_quality > 0.7
        }
        self.experience_memory.append(experience)
        
    def _hash_module_info(self, module_info: Dict[str, Any]) -> str:
        """Create a hash of module information for indexing."""
        key_features = {
            'classes': len(module_info.get('classes', [])),
            'functions': len(module_info.get('functions', [])),
            'complexity': module_info.get('complexity_score', 0),
            'imports': len(module_info.get('imports', []))
        }
        return hashlib.md5(str(sorted(key_features.items())).encode()).hexdigest()
    
    def _extract_features(self, module_info: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from module information."""
        features = [
            len(module_info.get('classes', [])),
            len(module_info.get('functions', [])),
            module_info.get('complexity_score', 0),
            len(module_info.get('imports', [])),
            int(module_info.get('has_main', False)),
            len(module_info.get('constants', []))
        ]
        return np.array(features, dtype=float)
    
    def predict_optimal_strategy(self, module_info: Dict[str, Any]) -> TestStrategy:
        """Predict optimal test strategy based on past experiences."""
        if not self.experience_memory:
            return TestStrategy.COMPREHENSIVE
            
        current_features = self._extract_features(module_info)
        
        # Find similar modules in experience
        similarities = []
        for exp in self.experience_memory:
            similarity = self._calculate_similarity(current_features, exp['module_features'])
            similarities.append((similarity, exp))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Weight strategies by similarity and success
        strategy_scores = defaultdict(float)
        total_weight = 0
        
        for similarity, exp in similarities[:10]:  # Top 10 similar experiences
            if exp['success']:
                weight = similarity * exp['quality_score']
                strategy_scores[exp['strategy']] += weight
                total_weight += weight
        
        if total_weight > 0:
            # Return strategy with highest weighted score
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            return best_strategy
        
        return TestStrategy.COMPREHENSIVE
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between feature vectors."""
        # Normalize features
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Cosine similarity
        return np.dot(features1, features2) / (norm1 * norm2)
    
    def adapt_generation_parameters(self, module_info: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt test generation parameters based on learning."""
        params = {
            'coverage_target': 0.8,
            'complexity_threshold': 10,
            'max_test_functions': 20,
            'include_edge_cases': True,
            'mock_external_deps': True
        }
        
        # Adjust based on module complexity
        complexity = module_info.get('complexity_score', 0)
        if complexity > 50:
            params['max_test_functions'] = 30
            params['complexity_threshold'] = 15
        elif complexity < 10:
            params['max_test_functions'] = 10
            params['coverage_target'] = 0.9
            
        # Adjust based on historical performance
        module_hash = self._hash_module_info(module_info)
        if module_hash in self.pattern_cache:
            cached_params = self.pattern_cache[module_hash]
            # Blend cached and default parameters
            for key in params:
                if key in cached_params:
                    params[key] = (params[key] + cached_params[key]) / 2
                    
        return params
    
    def update_learning(self, module_info: Dict[str, Any], 
                       strategy: TestStrategy, 
                       actual_performance: float):
        """Update learning based on actual test performance."""
        module_hash = self._hash_module_info(module_info)
        
        # Update pattern cache
        if module_hash not in self.pattern_cache:
            self.pattern_cache[module_hash] = {}
            
        # Record adaptation
        self.adaptation_history[module_hash].append({
            'strategy': strategy,
            'performance': actual_performance,
            'timestamp': time.time()
        })
        
        # Update performance metrics
        self.performance_metrics[strategy.value] = (
            self.performance_metrics[strategy.value] * 0.9 + 
            actual_performance * 0.1
        )
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process."""
        return {
            'total_experiences': len(self.experience_memory),
            'strategy_performance': dict(self.performance_metrics),
            'adaptation_count': sum(len(history) for history in self.adaptation_history.values()),
            'pattern_cache_size': len(self.pattern_cache),
            'learning_maturity': min(len(self.experience_memory) / 1000, 1.0)
        }
    
    def save_knowledge(self, filepath: Path):
        """Save learned knowledge to disk."""
        knowledge = {
            'experience_memory': list(self.experience_memory),
            'pattern_cache': self.pattern_cache,
            'adaptation_history': dict(self.adaptation_history),
            'performance_metrics': dict(self.performance_metrics)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(knowledge, f)
    
    def load_knowledge(self, filepath: Path):
        """Load previously learned knowledge."""
        if filepath.exists():
            with open(filepath, 'rb') as f:
                knowledge = pickle.load(f)
                self.experience_memory = deque(knowledge['experience_memory'], 
                                             maxlen=self.memory_size)
                self.pattern_cache = knowledge['pattern_cache']
                self.adaptation_history = defaultdict(list, knowledge['adaptation_history'])
                self.performance_metrics = defaultdict(float, knowledge['performance_metrics'])


class BaseTestAnalyzer(ABC):
    """Abstract base class for test analyzers."""
    
    @abstractmethod
    def analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a module to understand its structure and functionality."""
        pass
    
    @abstractmethod
    def extract_testable_units(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract testable units (functions, classes, methods) from analysis."""
        pass


class ASTAnalyzer(BaseTestAnalyzer):
    """Analyze modules using Abstract Syntax Tree parsing."""
    
    def analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a module using AST to understand its structure."""
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        content = module_path.read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"error": f"Syntax error in module: {e}"}
        
        analysis = {
            "module_name": module_path.stem,
            "file_path": str(module_path),
            "classes": [],
            "functions": [],
            "imports": [],
            "constants": [],
            "decorators": set(),
            "has_main": False,
            "docstring": ast.get_docstring(tree),
            "complexity_score": 0
        }
        
        # Analyze all nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node)
                analysis["classes"].append(class_info)
                analysis["complexity_score"] += class_info.get("complexity", 0)
            
            elif isinstance(node, ast.FunctionDef):
                # Only top-level functions (not methods)
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                    func_info = self._analyze_function(node)
                    analysis["functions"].append(func_info)
                    analysis["complexity_score"] += func_info.get("complexity", 0)
            
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                analysis["imports"].append(self._get_import_info(node))
            
            elif isinstance(node, ast.Assign):
                # Track module-level constants
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        analysis["constants"].append(target.id)
        
        # Check for if __name__ == "__main__"
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if self._is_main_check(node):
                    analysis["has_main"] = True
                    break
        
        return analysis
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition."""
        class_info = {
            "name": node.name,
            "methods": [],
            "properties": [],
            "class_variables": [],
            "docstring": ast.get_docstring(node),
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
            "base_classes": [self._get_base_class_name(base) for base in node.bases],
            "complexity": len(node.body)
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item)
                
                # Identify special methods
                if item.name.startswith("__") and item.name.endswith("__"):
                    method_info["type"] = "magic"
                elif item.name.startswith("_"):
                    method_info["type"] = "private"
                else:
                    method_info["type"] = "public"
                
                # Check if it's a property
                if any(d for d in method_info.get("decorators", []) if "property" in str(d)):
                    class_info["properties"].append(method_info)
                else:
                    class_info["methods"].append(method_info)
            
            elif isinstance(item, ast.Assign):
                # Class variables
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info["class_variables"].append(target.id)
        
        return class_info
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition."""
        return {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "defaults": len(node.args.defaults),
            "kwargs": node.args.kwarg.arg if node.args.kwarg else None,
            "varargs": node.args.vararg.arg if node.args.vararg else None,
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "returns_annotation": self._get_annotation(node.returns),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "complexity": len(list(ast.walk(node)))
        }
    
    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
        return str(decorator)
    
    def _get_base_class_name(self, base) -> str:
        """Extract base class name."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else str(base)
        return str(base)
    
    def _get_annotation(self, annotation) -> Optional[str]:
        """Extract type annotation."""
        if annotation is None:
            return None
        if isinstance(annotation, ast.Name):
            return annotation.id
        return str(annotation)
    
    def _get_import_info(self, node) -> Dict[str, Any]:
        """Extract import information."""
        if isinstance(node, ast.Import):
            return {
                "type": "import",
                "names": [alias.name for alias in node.names]
            }
        else:  # ImportFrom
            return {
                "type": "from",
                "module": node.module,
                "names": [alias.name for alias in node.names],
                "level": node.level
            }
    
    def _is_main_check(self, node: ast.If) -> bool:
        """Check if this is a if __name__ == "__main__" block."""
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                if len(node.test.comparators) == 1:
                    comp = node.test.comparators[0]
                    if isinstance(comp, ast.Constant) and comp.value == "__main__":
                        return True
        return False
    
    def extract_testable_units(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all testable units from the analysis."""
        units = []
        
        # Add standalone functions
        for func in analysis.get("functions", []):
            units.append({
                "type": "function",
                "name": func["name"],
                "module": analysis["module_name"],
                "details": func
            })
        
        # Add class methods
        for cls in analysis.get("classes", []):
            # Add class constructor test
            units.append({
                "type": "class",
                "name": cls["name"],
                "module": analysis["module_name"],
                "details": cls
            })
            
            # Add individual method tests
            for method in cls.get("methods", []):
                if method["name"] not in ["__init__", "__str__", "__repr__"]:
                    units.append({
                        "type": "method",
                        "name": f"{cls['name']}.{method['name']}",
                        "class": cls["name"],
                        "module": analysis["module_name"],
                        "details": method
                    })
        
        return units


class TestGenerator:
    """Generate test code based on analysis."""
    
    def generate_test_suite(self, analysis: Dict[str, Any], 
                           strategy: TestStrategy = TestStrategy.COMPREHENSIVE) -> str:
        """Generate a complete test suite for the analyzed module."""
        module_name = analysis["module_name"]
        
        # Start building the test file
        test_code = [
            '"""',
            f'Test suite for {module_name} module',
            'Generated by Unified Intelligent Test Builder',
            '"""',
            '',
            'import pytest',
            'import unittest',
            'from unittest.mock import Mock, patch, MagicMock',
            f'from {module_name} import *',
            '',
            ''
        ]
        
        # Generate tests for functions
        for func in analysis.get("functions", []):
            test_code.append(self._generate_function_test(func, module_name))
            test_code.append('')
        
        # Generate tests for classes
        for cls in analysis.get("classes", []):
            test_code.append(self._generate_class_test(cls, module_name))
            test_code.append('')
        
        return '\n'.join(test_code)
    
    def _generate_function_test(self, func: Dict[str, Any], module_name: str) -> str:
        """Generate test for a function."""
        func_name = func["name"]
        test_name = f"test_{func_name}"
        
        test_lines = [
            f'class Test{func_name.title()}:',
            f'    """Tests for {func_name} function."""',
            '',
            '    def setup_method(self):',
            '        """Set up test fixtures."""',
            '        pass',
            '',
            f'    def {test_name}_basic(self):',
            f'        """Test basic functionality of {func_name}."""',
            '        # TODO: Implement basic test',
            f'        # result = {func_name}()',
            '        # assert result is not None',
            '        pass',
            ''
        ]
        
        # Add parameter tests if function has parameters
        if func.get("args"):
            test_lines.extend([
                f'    def {test_name}_with_parameters(self):',
                f'        """Test {func_name} with various parameters."""',
                '        # TODO: Test with different parameter values',
                '        pass',
                ''
            ])
        
        # Add edge case tests
        test_lines.extend([
            f'    def {test_name}_edge_cases(self):',
            f'        """Test {func_name} with edge cases."""',
            '        # TODO: Test boundary conditions',
            '        # Test None inputs',
            '        # Test empty inputs',
            '        pass',
            ''
        ])
        
        # Add exception tests
        test_lines.extend([
            f'    def {test_name}_exceptions(self):',
            f'        """Test {func_name} exception handling."""',
            '        # TODO: Test error conditions',
            '        # with pytest.raises(ExpectedException):',
            f'        #     {func_name}(invalid_input)',
            '        pass'
        ])
        
        return '\n'.join(test_lines)
    
    def _generate_class_test(self, cls: Dict[str, Any], module_name: str) -> str:
        """Generate test for a class."""
        class_name = cls["name"]
        
        test_lines = [
            f'class Test{class_name}:',
            f'    """Tests for {class_name} class."""',
            '',
            '    def setup_method(self):',
            '        """Set up test fixtures."""',
            f'        self.instance = {class_name}()',
            '',
            f'    def test_{class_name.lower()}_initialization(self):',
            f'        """Test {class_name} initialization."""',
            '        assert self.instance is not None',
            f'        assert isinstance(self.instance, {class_name})',
            ''
        ]
        
        # Add tests for each public method
        for method in cls.get("methods", []):
            if method.get("type") == "public":
                method_name = method["name"]
                test_lines.extend([
                    f'    def test_{method_name}(self):',
                    f'        """Test {class_name}.{method_name} method."""',
                    '        # TODO: Implement method test',
                    f'        # result = self.instance.{method_name}()',
                    '        # assert result is not None',
                    '        pass',
                    ''
                ])
        
        # Add property tests
        for prop in cls.get("properties", []):
            prop_name = prop["name"]
            test_lines.extend([
                f'    def test_{prop_name}_property(self):',
                f'        """Test {class_name}.{prop_name} property."""',
                '        # TODO: Test property getter/setter',
                f'        # value = self.instance.{prop_name}',
                '        # assert value is not None',
                '        pass',
                ''
            ])
        
        return '\n'.join(test_lines)


class UnifiedTestBuilder:
    """
    Unified Intelligent Test Builder combining all capabilities.
    
    This class consolidates functionality from:
    - intelligent_test_builder.py (AI-powered generation)
    - intelligent_test_builder_offline.py (offline template generation)
    - intelligent_test_builder_v2.py (direct SDK integration)
    """
    
    def __init__(self, mode: TestBuilderMode = TestBuilderMode.OFFLINE, 
                 enable_meta_learning: bool = True):
        """
        Initialize the unified test builder.
        
        Args:
            mode: The operation mode (AI_POWERED, OFFLINE, or HYBRID)
            enable_meta_learning: Enable meta-learning capabilities
        """
        self.mode = mode
        self.analyzer = ASTAnalyzer()
        self.generator = TestGenerator()
        
        # Initialize meta-learning engine
        if enable_meta_learning:
            self.meta_learner = MetaLearningEngine()
            self._load_previous_knowledge()
        else:
            self.meta_learner = None
        
        # Initialize AI components if needed
        if mode in [TestBuilderMode.AI_POWERED, TestBuilderMode.HYBRID]:
            self._init_ai_components()
    
    def _init_ai_components(self):
        """Initialize AI components for powered modes."""
        try:
            # Try to import Gemini provider
            from multi_coder_analysis.llm_providers.gemini_provider import GeminiProvider
            self.ai_provider = GeminiProvider()
            print("AI components initialized successfully")
        except ImportError:
            print("Warning: AI components not available, falling back to offline mode")
            self.mode = TestBuilderMode.OFFLINE
    
    def _load_previous_knowledge(self):
        """Load previously learned knowledge if available."""
        if self.meta_learner:
            knowledge_path = Path(__file__).parent / "meta_learning_knowledge.pkl"
            try:
                self.meta_learner.load_knowledge(knowledge_path)
                print(f"Loaded meta-learning knowledge: {self.meta_learner.get_learning_insights()}")
            except (FileNotFoundError, Exception):
                print("No previous meta-learning knowledge found, starting fresh")
    
    def _save_knowledge(self):
        """Save accumulated knowledge to disk."""
        if self.meta_learner:
            knowledge_path = Path(__file__).parent / "meta_learning_knowledge.pkl"
            self.meta_learner.save_knowledge(knowledge_path)
    
    def build_tests(self, module_path: Path, 
                    strategy: TestStrategy = TestStrategy.COMPREHENSIVE,
                    output_path: Optional[Path] = None) -> str:
        """
        Build tests for a module with meta-learning enhancement.
        
        Args:
            module_path: Path to the module to test
            strategy: Test generation strategy
            output_path: Optional path to save the generated tests
        
        Returns:
            Generated test code as a string
        """
        start_time = time.time()
        
        # Analyze the module
        print(f"Analyzing module: {module_path}")
        analysis = self.analyzer.analyze_module(module_path)
        
        if "error" in analysis:
            raise ValueError(f"Module analysis failed: {analysis['error']}")
        
        # Use meta-learning to optimize strategy if available
        if self.meta_learner:
            predicted_strategy = self.meta_learner.predict_optimal_strategy(analysis)
            if predicted_strategy != strategy:
                print(f"Meta-learner suggests {predicted_strategy.value} instead of {strategy.value}")
                # Use predicted strategy if confidence is high
                insights = self.meta_learner.get_learning_insights()
                if insights['learning_maturity'] > 0.3:  # 30% learning maturity threshold
                    strategy = predicted_strategy
                    print(f"Adopting meta-learner suggestion: {strategy.value}")
            
            # Get adaptive parameters
            adaptive_params = self.meta_learner.adapt_generation_parameters(analysis)
            print(f"Adaptive parameters: {adaptive_params}")
        
        # Extract testable units
        units = self.analyzer.extract_testable_units(analysis)
        print(f"Found {len(units)} testable units")
        
        # Generate tests based on mode
        if self.mode == TestBuilderMode.OFFLINE:
            test_code = self.generator.generate_test_suite(analysis, strategy)
        elif self.mode == TestBuilderMode.AI_POWERED:
            test_code = self._generate_ai_powered_tests(analysis, strategy)
        else:  # HYBRID
            # Start with offline template, enhance with AI
            test_code = self.generator.generate_test_suite(analysis, strategy)
            test_code = self._enhance_with_ai(test_code, analysis)
        
        # Calculate execution time and quality metrics
        execution_time = time.time() - start_time
        test_quality = self._evaluate_test_quality(test_code)
        
        # Record experience for meta-learning
        if self.meta_learner:
            self.meta_learner.record_experience(
                analysis, strategy, test_quality, execution_time
            )
            
            # Update learning based on performance
            self.meta_learner.update_learning(analysis, strategy, test_quality)
            
            # Save knowledge periodically
            self._save_knowledge()
        
        # Save if output path provided
        if output_path:
            output_path.write_text(test_code)
            print(f"Tests saved to: {output_path}")
        
        print(f"Test generation completed in {execution_time:.2f}s with quality score {test_quality:.2f}")
        
        return test_code
    
    def _generate_ai_powered_tests(self, analysis: Dict[str, Any], 
                                  strategy: TestStrategy) -> str:
        """Generate tests using AI."""
        # This would use the AI provider to generate tests
        # For now, fall back to template generation
        print("AI-powered generation not fully implemented, using templates")
        return self.generator.generate_test_suite(analysis, strategy)
    
    def _evaluate_test_quality(self, test_code: str) -> float:
        """
        Evaluate the quality of generated test code.
        Agent C Phase 2 Enhancement: Meta-Learning Quality Assessment
        
        Returns a quality score between 0.0 and 1.0.
        """
        score = 0.0
        factors = 0
        
        # Check for syntax validity
        try:
            ast.parse(test_code)
            score += 0.3  # 30% for valid syntax
        except SyntaxError:
            pass
        factors += 1
        
        # Check for test framework usage
        if any(framework in test_code for framework in ["import pytest", "import unittest", "from unittest"]):
            score += 0.15  # 15% for proper framework usage
        factors += 1
        
        # Check for test function presence
        test_functions = len([line for line in test_code.split('\n') if 'def test_' in line])
        if test_functions > 0:
            score += 0.2  # 20% for having test functions
            # Bonus for multiple tests
            score += min(0.1, test_functions * 0.02)  # Up to 10% bonus
        factors += 1
        
        # Check for assertions
        assertion_count = test_code.count('assert')
        if assertion_count > 0:
            score += 0.15  # 15% for having assertions
            # Bonus for multiple assertions
            score += min(0.1, assertion_count * 0.01)  # Up to 10% bonus
        factors += 1
        
        # Check for docstrings and comments
        if '"""' in test_code or "'''" in test_code:
            score += 0.05  # 5% for documentation
        factors += 1
        
        # Check for proper imports
        import_lines = [line.strip() for line in test_code.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
        if len(import_lines) > 0:
            score += 0.05  # 5% for having imports
        factors += 1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the meta-learning system."""
        if self.meta_learner:
            return self.meta_learner.get_learning_insights()
        return {"meta_learning": "disabled"}
    
    def _enhance_with_ai(self, test_code: str, analysis: Dict[str, Any]) -> str:
        """Enhance template tests with AI suggestions."""
        # This would use AI to improve the template tests
        # For now, return as-is
        return test_code
    
    def validate_tests(self, test_code: str) -> Tuple[bool, List[str]]:
        """
        Validate generated test code.
        
        Args:
            test_code: The test code to validate
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Try to parse the test code
            ast.parse(test_code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors
        
        # Check for basic test structure
        if "import pytest" not in test_code and "import unittest" not in test_code:
            errors.append("No test framework imports found")
        
        if not any(word in test_code for word in ["def test_", "class Test"]):
            errors.append("No test functions or classes found")
        
        return len(errors) == 0, errors


def main():
    """Main entry point for the unified test builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Intelligent Test Builder")
    parser.add_argument("module", type=str, help="Path to the module to test")
    parser.add_argument("--mode", type=str, choices=["ai", "offline", "hybrid"],
                       default="offline", help="Test generation mode")
    parser.add_argument("--strategy", type=str, 
                       choices=["comprehensive", "quick", "focused", "regression"],
                       default="comprehensive", help="Test generation strategy")
    parser.add_argument("--output", type=str, help="Output path for generated tests")
    parser.add_argument("--disable-meta-learning", action="store_true", 
                       help="Disable meta-learning capabilities")
    parser.add_argument("--show-insights", action="store_true",
                       help="Show meta-learning insights")
    
    args = parser.parse_args()
    
    # Map string mode to enum
    mode_map = {
        "ai": TestBuilderMode.AI_POWERED,
        "offline": TestBuilderMode.OFFLINE,
        "hybrid": TestBuilderMode.HYBRID
    }
    mode = mode_map[args.mode]
    
    # Map string strategy to enum
    strategy_map = {
        "comprehensive": TestStrategy.COMPREHENSIVE,
        "quick": TestStrategy.QUICK,
        "focused": TestStrategy.FOCUSED,
        "regression": TestStrategy.REGRESSION
    }
    strategy = strategy_map[args.strategy]
    
    # Build tests
    module_path = Path(args.module)
    output_path = Path(args.output) if args.output else None
    
    builder = UnifiedTestBuilder(mode=mode, enable_meta_learning=not args.disable_meta_learning)
    
    # Show meta-learning insights if requested
    if args.show_insights:
        insights = builder.get_meta_learning_insights()
        print("\n📊 Meta-Learning Insights:")
        for key, value in insights.items():
            print(f"  {key}: {value}")
        print()
    
    try:
        test_code = builder.build_tests(module_path, strategy, output_path)
        
        # Validate the generated tests
        is_valid, errors = builder.validate_tests(test_code)
        
        if is_valid:
            print("✅ Tests generated successfully!")
        else:
            print("⚠️ Tests generated with warnings:")
            for error in errors:
                print(f"  - {error}")
        
        if not output_path:
            print("\nGenerated test preview:")
            print("-" * 50)
            print(test_code[:1000])
            if len(test_code) > 1000:
                print("... (truncated)")
    
    except Exception as e:
        print(f"❌ Error generating tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()