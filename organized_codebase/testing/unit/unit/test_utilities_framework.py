from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
TestMaster Utilities Framework Test Suite

Comprehensive tests for the unified utilities framework to ensure
zero functionality loss during consolidation.
"""

import ast
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import all utilities to test
from . import (
    MLIssue, AnalysisIssue, ML_FRAMEWORKS, ML_ANTIPATTERNS,
    extract_tensor_shapes, detect_ml_frameworks_in_content,
    analyze_import_patterns, check_version_compatibility,
    extract_hyperparameters, identify_model_patterns,
    extract_common_patterns, calculate_complexity_score,
    AnalysisEngine, BusinessAnalysisUtils, SemanticAnalysisUtils,
    DebtAnalysisUtils, MetaprogAnalysisUtils, EnergyAnalysisUtils,
    WebMonitoringUtils, DashboardGenerator,
    UtilityExtractor, FunctionAnalyzer
)


class TestCoreUtilities(unittest.TestCase):
    """Test core utility functions"""
    
    def test_ml_issue_creation(self):
        """Test MLIssue data structure"""
        issue = MLIssue(
            type="data_leakage",
            severity="high",
            location="line 42",
            description="Test issue",
            recommendation="Fix it",
            framework="tensorflow",
            impact="high"
        )
        
        self.assertEqual(issue.type, "data_leakage")
        self.assertEqual(issue.framework, "tensorflow")
    
    def test_analysis_issue_creation(self):
        """Test AnalysisIssue data structure"""
        issue = AnalysisIssue(
            issue_type="naming",
            severity="medium",
            location="line 10",
            description="Poor naming",
            recommendation="Use better names",
            impact="readability"
        )
        
        self.assertEqual(issue.issue_type, "naming")
        self.assertEqual(issue.severity, "medium")
    
    def test_ml_frameworks_patterns(self):
        """Test ML frameworks detection patterns"""
        self.assertIn("tensorflow", ML_FRAMEWORKS)
        self.assertIn("pytorch", ML_FRAMEWORKS)
        self.assertIn("import tensorflow", ML_FRAMEWORKS["tensorflow"])
        self.assertIn("import torch", ML_FRAMEWORKS["pytorch"])
    
    def test_detect_ml_frameworks_in_content(self):
        """Test ML framework detection in code content"""
        content = """
        import tensorflow as tf
        import torch
        from sklearn import datasets
        """
        
        detected = detect_ml_frameworks_in_content(content)
        
        self.assertIn("tensorflow", detected)
        self.assertIn("pytorch", detected)
        self.assertIn("sklearn", detected)
    
    def test_extract_tensor_shapes(self):
        """Test tensor shape extraction"""
        code = """
        x.reshape(28, 28, 1)
        y.view(-1, 784)
        z.shape = [32, 64]
        """
        
        shapes = extract_tensor_shapes(code)
        
        self.assertTrue(len(shapes) > 0)
        self.assertTrue(any("reshape" in shape["pattern"] for shape in shapes))
    
    def test_analyze_import_patterns(self):
        """Test import pattern analysis"""
        code = """
        import numpy as np
        from tensorflow import keras
        import torch
        from sklearn import *
        """
        
        tree = ast.parse(code)
        imports = analyze_import_patterns(tree)
        
        self.assertIn("alias_imports", imports)
        self.assertIn("from_imports", imports)
        self.assertIn("star_imports", imports)
        
        # Check that numpy import with alias is detected
        alias_imports = imports["alias_imports"]
        numpy_import = next((imp for imp in alias_imports if imp["module"] == "numpy"), None)
        self.assertIsNotNone(numpy_import)
        self.assertEqual(numpy_import["alias"], "np")
    
    def test_check_version_compatibility(self):
        """Test framework version compatibility checking"""
        frameworks = ["tensorflow", "pytorch"]
        conflicts = check_version_compatibility(frameworks)
        
        self.assertTrue(len(conflicts) > 0)
        self.assertTrue(any("Mixed TensorFlow and PyTorch" in conflict["issue"] for conflict in conflicts))
    
    def test_extract_hyperparameters(self):
        """Test hyperparameter extraction"""
        code = """
        learning_rate = 0.001
        batch_size = 32
        epochs = 100
        hidden_size = 256
        """
        
        tree = ast.parse(code)
        hyperparams = extract_hyperparameters(tree)
        
        self.assertTrue(len(hyperparams) >= 4)
        
        lr_param = next((hp for hp in hyperparams if hp["name"] == "learning_rate"), None)
        self.assertIsNotNone(lr_param)
        self.assertEqual(lr_param["value"], 0.001)
    
    def test_identify_model_patterns(self):
        """Test model pattern identification"""
        code = """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128)
        ])
        
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
        
        clf = RandomForestClassifier()
        """
        
        tree = ast.parse(code)
        patterns = identify_model_patterns(tree)
        
        self.assertTrue(len(patterns) > 0)
        frameworks = [p["framework"] for p in patterns]
        self.assertTrue(any("tensorflow" in fw for fw in frameworks))


class TestPatternUtilities(unittest.TestCase):
    """Test pattern analysis utilities"""
    
    def test_calculate_complexity_score(self):
        """Test complexity score calculation"""
        code = """
        def simple_function():
            return 42
        
        def complex_function(x):
            if x > 0:
                for i in range(x):
                    if i % 2 == 0:
                        try:
                            result = i * 2
                        except:
                            continue
            else:
                while x < 10:
                    x += 1
            return x
        """
        
        tree = ast.parse(code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        simple_func = functions[0]
        complex_func = functions[1]
        
        simple_complexity = calculate_complexity_score(simple_func)
        complex_complexity = calculate_complexity_score(complex_func)
        
        self.assertEqual(simple_complexity, 1)  # Base complexity
        self.assertGreater(complex_complexity, simple_complexity)
    
    def test_extract_common_patterns(self):
        """Test common pattern extraction"""
        code = """
        def calculate_roi():
            pass
        
        def process_payment():
            pass
        """
        
        tree = ast.parse(code)
        patterns = extract_common_patterns(tree, code)
        
        # Should find business analysis patterns
        business_patterns = [p for p in patterns if p["category"] == "analysis"]
        self.assertTrue(len(business_patterns) > 0)


class TestAnalysisUtilities(unittest.TestCase):
    """Test analysis-specific utilities"""
    
    def test_analysis_engine_base(self):
        """Test base analysis engine"""
        engine = AnalysisEngine()
        
        code = """
        def test_function():
            if True:
                return 42
        """
        
        tree = ast.parse(code)
        results = engine.analyze_tree(tree, code)
        
        self.assertIn("issues", results)
        self.assertIn("patterns", results)
        self.assertIn("metrics", results)
        self.assertIn("complexity", results["metrics"])
    
    def test_business_analysis_utils(self):
        """Test business analysis utilities"""
        analyzer = BusinessAnalysisUtils()
        
        code = """
        def process_payment(amount):
            print("Processing payment")  # I/O operation
            return amount * 0.95  # Business logic
        
        def calculate_discount():
            if amount > 100:  # Hardcoded business rule
                return 0.1
            return 0.05
        """
        
        tree = ast.parse(code)
        issues = analyzer.analyze_business_logic(tree, code)
        
        # Should find mixed concerns and hardcoded rules
        issue_types = [issue.issue_type for issue in issues]
        self.assertIn("mixed_concerns", issue_types)
    
    def test_debt_analysis_utils(self):
        """Test technical debt analysis utilities"""
        analyzer = DebtAnalysisUtils()
        
        code = """
        # TODO: Refactor this function
        # FIXME: This is broken
        # HACK: Temporary workaround
        def long_function(a, b, c, d, e, f, g, h):  # Too many parameters
            # Very long function with many statements
            x = a + b
            y = c + d
            z = e + f
            w = g + h
            # ... many more lines would be here
            return x + y + z + w
        """
        
        tree = ast.parse(code)
        issues = analyzer.analyze_technical_debt(tree, code)
        
        issue_types = [issue.issue_type for issue in issues]
        self.assertIn("todo_comment", issue_types)
        self.assertIn("fixme_comment", issue_types)
    
    def test_semantic_analysis_utils(self):
        """Test semantic analysis utilities"""
        analyzer = SemanticAnalysisUtils()
        
        code = """
        def a():  # Meaningless name
            pass
        
        def temp1():  # Generic name
            pass
        
        def calculate_user_discount():  # Good semantic name
            pass
        """
        
        tree = ast.parse(code)
        issues = analyzer.analyze_naming_semantics(tree)
        
        # Should find meaningless names
        issue_types = [issue.issue_type for issue in issues]
        self.assertIn("meaningless_name", issue_types)
    
    def test_energy_analysis_utils(self):
        """Test energy efficiency analysis"""
        analyzer = EnergyAnalysisUtils()
        
        code = """
        def inefficient_function():
            result = ""
            for i in range(1000):
                for j in range(1000):  # Nested loops
                    for k in range(10):  # Deep nesting
                        result += str(i)  # String concatenation in loop
            return result
        """
        
        tree = ast.parse(code)
        issues = analyzer.analyze_energy_efficiency(tree, code)
        
        issue_types = [issue.issue_type for issue in issues]
        self.assertIn("nested_loops", issue_types)


class TestWebUtilities(unittest.TestCase):
    """Test web monitoring utilities"""
    
    def test_web_monitoring_utils(self):
        """Test web monitoring utility functions"""
        # Test status indicator
        active_indicator = WebMonitoringUtils.get_status_indicator('active')
        self.assertIn('status-active', active_indicator)
        
        # Test success rate calculation
        rate = WebMonitoringUtils.calculate_success_rate(100, 85)
        self.assertEqual(rate, 85.0)
        
        # Test bytes formatting
        formatted = WebMonitoringUtils.format_bytes(1024)
        self.assertEqual(formatted, "1.0 KB")
        
        # Test HTML sanitization
        sanitized = WebMonitoringUtils.sanitize_html('<script>alert("test")</script>')
        self.assertNotIn('<script>', sanitized)
        self.assertIn('&lt;script&gt;', sanitized)
    
    def test_dashboard_generator(self):
        """Test dashboard generation"""
        generator = DashboardGenerator()
        
        # Test metrics card generation
        metrics = {
            'system': {'cpu_usage': 45.2, 'memory_usage': 67.8},
            'components': {'active_agents': 12, 'active_bridges': 3},
            'workflow': {'queue_size': 5, 'events_per_second': 2.1}
        }
        
        card = generator.create_metrics_card(metrics)
        self.assertIn('System Metrics', card)
        self.assertIn('45.2%', card)  # CPU usage
        self.assertIn('67.8%', card)  # Memory usage
        
        # Test component status card
        components = {
            'analytics_engine': 'active',
            'workflow_manager': 'inactive',
            'security_monitor': 'warning'
        }
        
        status_card = generator.create_component_status_card(components)
        self.assertIn('Component Status', status_card)
        self.assertIn('Analytics Engine', status_card)


class TestToolUtilities(unittest.TestCase):
    """Test utility extraction and analysis tools"""
    
    def test_utility_extractor(self):
        """Test utility function extraction"""
        extractor = UtilityExtractor()
        
        # Create a temporary test file content
        test_code = """
        def get_user_name(user_id):
            '''Get user name by ID'''
            return f"user_{user_id}"
        
        def calculate_total(items):
            '''Calculate total from items'''
            return sum(item.price for item in items)
        
        class DataProcessor:
            def process_data(self, data):
                return data.upper()
        
        def complex_function(a, b, c):
            if a > b:
                for i in range(c):
                    if i % 2:
                        result = a * b * i
                        try:
                            return result / c
                        except ZeroDivisionError:
                            continue
            return 0
        """
        
        tree = ast.parse(test_code)
        functions = extractor._extract_functions(tree, "test.py")
        
        self.assertEqual(len(functions), 4)  # 3 functions + 1 method
        
        # Check utility function detection
        utility_functions = [f for f in functions if f['is_utility']]
        self.assertTrue(len(utility_functions) >= 2)  # get_user_name and calculate_total
        
        # Check complexity calculation
        complex_func = next(f for f in functions if f['name'] == 'complex_function')
        self.assertGreater(complex_func['complexity'], 5)
    
    def test_function_analyzer(self):
        """Test function analysis"""
        analyzer = FunctionAnalyzer()
        
        # Mock functions data
        functions = [
            {
                'name': 'get_user_data',
                'args': ['user_id'],
                'is_utility': True,
                'complexity': 3,
                'docstring': 'Get user data'
            },
            {
                'name': 'get_user_profile', 
                'args': ['user_id'],
                'is_utility': True,
                'complexity': 4,
                'docstring': 'Get user profile'
            },
            {
                'name': 'complex_business_logic',
                'args': ['a', 'b', 'c', 'd'],
                'is_utility': False,
                'complexity': 15,
                'docstring': None
            }
        ]
        
        analysis = analyzer.analyze_function_patterns(functions)
        
        self.assertEqual(analysis['total_functions'], 3)
        self.assertEqual(analysis['utility_functions'], 2)
        self.assertEqual(analysis['complex_functions'], 1)
        
        # Should find similar functions (get_user_data and get_user_profile)
        similar_groups = analysis['similar_functions']
        self.assertTrue(len(similar_groups) > 0)


class TestFrameworkIntegration(unittest.TestCase):
    """Test framework integration and compatibility"""
    
    def test_all_imports_work(self):
        """Test that all imports work correctly"""
        from TestMaster.utilities import (
            MLIssue, AnalysisIssue, AnalysisEngine,
            WebMonitoringUtils, UtilityExtractor
        )
        
        # Should not raise any import errors
        self.assertTrue(True)
    
    def test_zero_functionality_loss(self):
        """Test that consolidation preserves all functionality"""
        # Test that all major functions from original files are available
        
        # ML utilities
        self.assertTrue(callable(detect_ml_frameworks_in_content))
        self.assertTrue(callable(extract_tensor_shapes))
        self.assertTrue(callable(analyze_import_patterns))
        
        # Pattern utilities (originally stubbed, now implemented)
        self.assertTrue(callable(extract_common_patterns))
        self.assertTrue(callable(calculate_complexity_score))
        
        # Analysis utilities (originally stubbed, now implemented)
        analyzer = BusinessAnalysisUtils()
        self.assertTrue(hasattr(analyzer, 'analyze_business_logic'))
        
        # Web utilities
        self.assertTrue(callable(WebMonitoringUtils.get_status_indicator))
        
        # Tool utilities
        extractor = UtilityExtractor()
        self.assertTrue(hasattr(extractor, 'extract_utilities_from_file'))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)