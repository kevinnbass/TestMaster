"""
ML-Powered Test Quality Scoring Framework for TestMaster
Analyzes and scores test quality using multiple metrics
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import math


class QualityMetric(Enum):
    """Test quality metrics"""
    COVERAGE = "coverage"
    ASSERTIONS = "assertions"
    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    ISOLATION = "isolation"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"


@dataclass
class TestMetrics:
    """Metrics for a test case"""
    name: str
    lines_of_code: int
    num_assertions: int
    assertion_quality: float
    cyclomatic_complexity: int
    coverage_percentage: float
    setup_teardown: bool
    mocking_used: bool
    documentation_score: float
    execution_time: Optional[float] = None
    failure_rate: Optional[float] = None


@dataclass
class QualityScore:
    """Test quality score with breakdown"""
    overall_score: float  # 0-100
    metric_scores: Dict[QualityMetric, float]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class TestAnalyzer:
    """Analyzes test code structure and quality"""
    
    def analyze_test_code(self, test_code: str) -> TestMetrics:
        """Analyze test code and extract metrics"""
        try:
            tree = ast.parse(test_code)
        except SyntaxError:
            return self._empty_metrics()
        
        metrics = TestMetrics(
            name=self._extract_test_name(tree),
            lines_of_code=len(test_code.splitlines()),
            num_assertions=self._count_assertions(tree),
            assertion_quality=self._analyze_assertion_quality(tree),
            cyclomatic_complexity=self._calculate_complexity(tree),
            coverage_percentage=0.0,  # Would need coverage data
            setup_teardown=self._has_setup_teardown(tree),
            mocking_used=self._uses_mocking(tree),
            documentation_score=self._score_documentation(tree, test_code)
        )
        
        return metrics
    
    def _extract_test_name(self, tree: ast.AST) -> str:
        """Extract test class/function name"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if 'test' in node.name.lower():
                    return node.name
            elif isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    return node.name
        return "unknown_test"
    
    def _count_assertions(self, tree: ast.AST) -> int:
        """Count assertion statements"""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'attr'):
                    # unittest assertions
                    if node.func.attr.startswith('assert'):
                        count += 1
                elif hasattr(node.func, 'id'):
                    # pytest assertions
                    if node.func.id == 'assert':
                        count += 1
            elif isinstance(node, ast.Assert):
                count += 1
        return count
    
    def _analyze_assertion_quality(self, tree: ast.AST) -> float:
        """Analyze quality of assertions"""
        quality_score = 0.0
        assertion_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, 'attr'):
                attr = node.func.attr
                
                # High quality assertions
                if attr in ['assertEqual', 'assertAlmostEqual', 'assertIn', 
                           'assertRaises', 'assertRegex']:
                    quality_score += 1.0
                    assertion_count += 1
                    
                # Medium quality assertions
                elif attr in ['assertTrue', 'assertFalse', 'assertIsNone',
                             'assertIsNotNone']:
                    quality_score += 0.5
                    assertion_count += 1
                    
                # Low quality assertions
                elif attr == 'assert':
                    quality_score += 0.3
                    assertion_count += 1
                    
        return (quality_score / assertion_count) if assertion_count > 0 else 0.0
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
                
        return complexity
    
    def _has_setup_teardown(self, tree: ast.AST) -> bool:
        """Check for setup/teardown methods"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in ['setUp', 'tearDown', 'setup', 'teardown',
                                'setup_method', 'teardown_method']:
                    return True
        return False
    
    def _uses_mocking(self, tree: ast.AST) -> bool:
        """Check if test uses mocking"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if 'mock' in alias.name.lower():
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and 'mock' in node.module.lower():
                    return True
        return False
    
    def _score_documentation(self, tree: ast.AST, code: str) -> float:
        """Score test documentation"""
        score = 0.0
        
        # Check for docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if ast.get_docstring(node):
                    score += 0.3
                    
        # Check for comments
        comment_lines = len([l for l in code.splitlines() if l.strip().startswith('#')])
        total_lines = len(code.splitlines())
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        
        score += min(comment_ratio * 2, 0.4)  # Cap at 0.4
        
        # Check for descriptive test names
        if re.search(r'test_\w+_when_\w+_then_\w+', code):
            score += 0.3
        elif re.search(r'test_\w+_\w+', code):
            score += 0.2
            
        return min(score, 1.0)
    
    def _empty_metrics(self) -> TestMetrics:
        """Return empty metrics for invalid code"""
        return TestMetrics(
            name="invalid",
            lines_of_code=0,
            num_assertions=0,
            assertion_quality=0.0,
            cyclomatic_complexity=0,
            coverage_percentage=0.0,
            setup_teardown=False,
            mocking_used=False,
            documentation_score=0.0
        )


class TestQualityScorer:
    """ML-powered test quality scoring engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analyzer = TestAnalyzer()
        self.weights = self._initialize_weights()
        
    def _initialize_weights(self) -> Dict[QualityMetric, float]:
        """Initialize metric weights for scoring"""
        return {
            QualityMetric.COVERAGE: 0.20,
            QualityMetric.ASSERTIONS: 0.20,
            QualityMetric.COMPLEXITY: 0.15,
            QualityMetric.MAINTAINABILITY: 0.15,
            QualityMetric.DOCUMENTATION: 0.10,
            QualityMetric.ISOLATION: 0.10,
            QualityMetric.PERFORMANCE: 0.05,
            QualityMetric.RELIABILITY: 0.05
        }
    
    def score_test(self, test_code: str, coverage_data: Optional[Dict] = None,
                  execution_data: Optional[Dict] = None) -> QualityScore:
        """Score test quality using ML model"""
        
        # Analyze test code
        metrics = self.analyzer.analyze_test_code(test_code)
        
        # Add external data if available
        if coverage_data:
            metrics.coverage_percentage = coverage_data.get('coverage', 0.0)
        if execution_data:
            metrics.execution_time = execution_data.get('time', None)
            metrics.failure_rate = execution_data.get('failure_rate', None)
        
        # Calculate individual metric scores
        metric_scores = self._calculate_metric_scores(metrics)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metric_scores)
        
        # Generate insights
        strengths = self._identify_strengths(metric_scores)
        weaknesses = self._identify_weaknesses(metric_scores)
        recommendations = self._generate_recommendations(metrics, metric_scores)
        
        return QualityScore(
            overall_score=overall_score,
            metric_scores=metric_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    def _calculate_metric_scores(self, metrics: TestMetrics) -> Dict[QualityMetric, float]:
        """Calculate individual metric scores"""
        scores = {}
        
        # Coverage score
        scores[QualityMetric.COVERAGE] = min(metrics.coverage_percentage, 100.0)
        
        # Assertion score
        assertion_density = metrics.num_assertions / max(metrics.lines_of_code, 1) * 100
        assertion_score = min(assertion_density * 10, 100) * metrics.assertion_quality
        scores[QualityMetric.ASSERTIONS] = assertion_score
        
        # Complexity score (lower is better)
        complexity_score = max(0, 100 - (metrics.cyclomatic_complexity - 1) * 10)
        scores[QualityMetric.COMPLEXITY] = complexity_score
        
        # Maintainability score
        maintainability = 0.0
        if metrics.lines_of_code < 50:
            maintainability += 40
        elif metrics.lines_of_code < 100:
            maintainability += 20
        if metrics.setup_teardown:
            maintainability += 30
        if metrics.mocking_used:
            maintainability += 30
        scores[QualityMetric.MAINTAINABILITY] = maintainability
        
        # Documentation score
        scores[QualityMetric.DOCUMENTATION] = metrics.documentation_score * 100
        
        # Isolation score
        isolation = 100.0 if metrics.mocking_used else 50.0
        scores[QualityMetric.ISOLATION] = isolation
        
        # Performance score
        if metrics.execution_time is not None:
            if metrics.execution_time < 0.1:
                performance = 100
            elif metrics.execution_time < 1.0:
                performance = 80
            elif metrics.execution_time < 5.0:
                performance = 60
            else:
                performance = 40
        else:
            performance = 50
        scores[QualityMetric.PERFORMANCE] = performance
        
        # Reliability score
        if metrics.failure_rate is not None:
            reliability = (1 - metrics.failure_rate) * 100
        else:
            reliability = 50
        scores[QualityMetric.RELIABILITY] = reliability
        
        return scores
    
    def _calculate_overall_score(self, metric_scores: Dict[QualityMetric, float]) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0
        
        for metric, score in metric_scores.items():
            weight = self.weights.get(metric, 0.0)
            total_score += score * weight
            
        return round(total_score, 2)
    
    def _identify_strengths(self, scores: Dict[QualityMetric, float]) -> List[str]:
        """Identify test strengths"""
        strengths = []
        
        for metric, score in scores.items():
            if score >= 80:
                strengths.append(f"Excellent {metric.value}: {score:.1f}%")
            elif score >= 70:
                strengths.append(f"Good {metric.value}: {score:.1f}%")
                
        return strengths[:5]  # Top 5 strengths
    
    def _identify_weaknesses(self, scores: Dict[QualityMetric, float]) -> List[str]:
        """Identify test weaknesses"""
        weaknesses = []
        
        for metric, score in scores.items():
            if score < 40:
                weaknesses.append(f"Poor {metric.value}: {score:.1f}%")
            elif score < 60:
                weaknesses.append(f"Below average {metric.value}: {score:.1f}%")
                
        return weaknesses[:5]  # Top 5 weaknesses
    
    def _generate_recommendations(self, metrics: TestMetrics, 
                                 scores: Dict[QualityMetric, float]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if scores[QualityMetric.ASSERTIONS] < 60:
            recommendations.append("Add more specific assertions to verify behavior")
            
        if scores[QualityMetric.COVERAGE] < 80:
            recommendations.append("Increase code coverage to at least 80%")
            
        if scores[QualityMetric.COMPLEXITY] < 70:
            recommendations.append("Simplify test logic to reduce complexity")
            
        if scores[QualityMetric.DOCUMENTATION] < 60:
            recommendations.append("Add docstrings and comments to explain test purpose")
            
        if not metrics.setup_teardown:
            recommendations.append("Add setup/teardown methods for better test isolation")
            
        if not metrics.mocking_used and scores[QualityMetric.ISOLATION] < 70:
            recommendations.append("Use mocking to isolate units under test")
            
        if metrics.lines_of_code > 100:
            recommendations.append("Split large test into smaller, focused tests")
            
        return recommendations[:5]  # Top 5 recommendations
    
    def train_model(self, training_data: List[Tuple[TestMetrics, float]]):
        """Train ML model on labeled test quality data"""
        # Placeholder for ML training
        # Would use sklearn or similar to train a model
        pass
    
    def batch_score(self, test_files: List[str]) -> Dict[str, QualityScore]:
        """Score multiple test files"""
        results = {}
        
        for file_path in test_files:
            try:
                with open(file_path, 'r') as f:
                    test_code = f.read()
                results[file_path] = self.score_test(test_code)
            except Exception as e:
                results[file_path] = QualityScore(
                    overall_score=0.0,
                    metric_scores={},
                    strengths=[],
                    weaknesses=[f"Error analyzing file: {e}"],
                    recommendations=[]
                )
                
        return results