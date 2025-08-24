"""
Quality Analyzer - Advanced code quality analysis and improvement recommendations

This module provides comprehensive code quality analysis capabilities including
maintainability assessment, readability improvements, design pattern recognition,
and automated refactoring suggestions with expert-level insights.

Key Capabilities:
- Cyclomatic and cognitive complexity analysis with recommendations
- Code maintainability index calculation and improvement guidance
- Design pattern recognition and anti-pattern detection
- Code duplication analysis with refactoring suggestions
- Documentation coverage assessment and enhancement recommendations
- Technical debt quantification and remediation strategies
"""

import ast
import re
import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path

from .optimization_models import (
    OptimizationType, OptimizationPriority, OptimizationStrategy,
    OptimizationRecommendation, QualityMetrics, RiskAssessment,
    create_optimization_recommendation, create_risk_assessment
)

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """Represents a detected code quality issue"""
    issue_type: str
    severity: str
    confidence: float
    description: str
    location: Tuple[int, int]
    affected_element: str
    code_snippet: str
    improvement_suggestion: str
    maintainability_impact: float = 0.0
    readability_impact: float = 0.0
    testability_impact: float = 0.0
    technical_debt_hours: float = 0.0


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for code elements"""
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    nesting_depth: int = 0
    method_count: int = 0
    line_count: int = 0
    parameter_count: int = 0
    maintainability_index: float = 100.0
    complexity_rating: str = "simple"


@dataclass
class DesignPattern:
    """Represents a detected or recommended design pattern"""
    pattern_name: str
    pattern_type: str
    confidence: float
    description: str
    benefits: List[str]
    implementation_guidance: str
    code_example: str = ""
    anti_pattern_detected: bool = False


class QualityAnalyzer:
    """
    Advanced code quality analysis engine
    
    Provides comprehensive quality analysis through static code analysis,
    complexity measurement, pattern detection, and improvement recommendations.
    """
    
    def __init__(self):
        """Initialize quality analyzer with analysis frameworks"""
        self.complexity_thresholds = {
            'cyclomatic_simple': 10,
            'cyclomatic_moderate': 20,
            'cyclomatic_complex': 30,
            'cognitive_simple': 15,
            'cognitive_moderate': 25,
            'nesting_depth': 4,
            'method_lines': 50,
            'class_methods': 20
        }
        
        self.quality_patterns = self._load_quality_patterns()
        self.design_patterns = self._load_design_patterns()
        self.refactoring_opportunities = {}
        
        logger.info("Quality Analyzer initialized")
    
    async def analyze_quality(self, code: str, tree: ast.AST, file_path: str = "") -> List[OptimizationRecommendation]:
        """
        Comprehensive quality analysis of code
        
        Args:
            code: Source code as string
            tree: AST representation of the code
            file_path: Path to the file being analyzed
            
        Returns:
            List of quality optimization recommendations
        """
        recommendations = []
        
        try:
            # Multi-layer quality analysis
            complexity_issues = await self._analyze_complexity(code, tree)
            maintainability_issues = await self._analyze_maintainability(code, tree)
            readability_issues = await self._analyze_readability(code, tree)
            documentation_issues = await self._analyze_documentation(code, tree)
            design_issues = await self._analyze_design_patterns(code, tree)
            duplication_issues = await self._analyze_code_duplication(code, tree)
            
            # Convert analysis results to recommendations
            recommendations.extend(self._create_complexity_recommendations(complexity_issues, file_path))
            recommendations.extend(self._create_maintainability_recommendations(maintainability_issues, file_path))
            recommendations.extend(self._create_readability_recommendations(readability_issues, file_path))
            recommendations.extend(self._create_documentation_recommendations(documentation_issues, file_path))
            recommendations.extend(self._create_design_recommendations(design_issues, file_path))
            recommendations.extend(self._create_duplication_recommendations(duplication_issues, file_path))
            
            # Apply quality-specific prioritization
            recommendations = self._prioritize_quality_recommendations(recommendations)
            
            logger.info(f"Generated {len(recommendations)} quality recommendations for {file_path}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in quality analysis: {e}")
            return []
    
    async def _analyze_complexity(self, code: str, tree: ast.AST) -> List[QualityIssue]:
        """Analyze code complexity metrics"""
        issues = []
        
        class ComplexityAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                self.current_class = None
            
            def visit_FunctionDef(self, node):
                self.current_function = node.name
                metrics = self._calculate_function_complexity(node)
                
                # Cyclomatic complexity check
                if metrics.cyclomatic_complexity > self.thresholds['cyclomatic_moderate']:
                    severity = "high" if metrics.cyclomatic_complexity > self.thresholds['cyclomatic_complex'] else "medium"
                    issues.append(QualityIssue(
                        issue_type="high_cyclomatic_complexity",
                        severity=severity,
                        confidence=0.9,
                        description=f"Function '{node.name}' has high cyclomatic complexity ({metrics.cyclomatic_complexity})",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        affected_element=node.name,
                        code_snippet=f"def {node.name}(...): # {metrics.cyclomatic_complexity} complexity",
                        improvement_suggestion="Consider breaking function into smaller, focused functions",
                        maintainability_impact=0.3 * (metrics.cyclomatic_complexity - 10) / 10,
                        testability_impact=0.4 * (metrics.cyclomatic_complexity - 10) / 10,
                        technical_debt_hours=max(1.0, (metrics.cyclomatic_complexity - 10) * 0.5)
                    ))
                
                # Cognitive complexity check
                if metrics.cognitive_complexity > self.thresholds['cognitive_moderate']:
                    issues.append(QualityIssue(
                        issue_type="high_cognitive_complexity",
                        severity="medium",
                        confidence=0.8,
                        description=f"Function '{node.name}' has high cognitive complexity ({metrics.cognitive_complexity})",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        affected_element=node.name,
                        code_snippet=f"def {node.name}(...): # cognitive load: {metrics.cognitive_complexity}",
                        improvement_suggestion="Reduce nested conditions and simplify logic flow",
                        readability_impact=0.4,
                        maintainability_impact=0.3,
                        technical_debt_hours=metrics.cognitive_complexity * 0.3
                    ))
                
                # Function length check
                if metrics.line_count > self.thresholds['method_lines']:
                    issues.append(QualityIssue(
                        issue_type="long_function",
                        severity="medium",
                        confidence=0.7,
                        description=f"Function '{node.name}' is too long ({metrics.line_count} lines)",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        affected_element=node.name,
                        code_snippet=f"def {node.name}(...): # {metrics.line_count} lines",
                        improvement_suggestion="Extract functionality into smaller, focused methods",
                        maintainability_impact=0.2,
                        readability_impact=0.3,
                        technical_debt_hours=max(1.0, (metrics.line_count - 50) * 0.1)
                    ))
                
                # Parameter count check
                if metrics.parameter_count > 5:
                    issues.append(QualityIssue(
                        issue_type="too_many_parameters",
                        severity="medium",
                        confidence=0.8,
                        description=f"Function '{node.name}' has too many parameters ({metrics.parameter_count})",
                        location=(node.lineno, node.lineno),
                        affected_element=node.name,
                        code_snippet=f"def {node.name}(...): # {metrics.parameter_count} parameters",
                        improvement_suggestion="Consider using parameter objects or data classes",
                        maintainability_impact=0.2,
                        testability_impact=0.3,
                        technical_debt_hours=metrics.parameter_count * 0.2
                    ))
                
                self.generic_visit(node)
                self.current_function = None
            
            def visit_ClassDef(self, node):
                self.current_class = node.name
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                
                # Too many methods check
                if len(methods) > self.thresholds['class_methods']:
                    issues.append(QualityIssue(
                        issue_type="god_class",
                        severity="high",
                        confidence=0.8,
                        description=f"Class '{node.name}' has too many methods ({len(methods)})",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        affected_element=node.name,
                        code_snippet=f"class {node.name}: # {len(methods)} methods",
                        improvement_suggestion="Consider splitting into smaller, more focused classes",
                        maintainability_impact=0.4,
                        testability_impact=0.3,
                        technical_debt_hours=max(2.0, len(methods) * 0.3)
                    ))
                
                self.generic_visit(node)
                self.current_class = None
            
            def _calculate_function_complexity(self, node: ast.FunctionDef) -> ComplexityMetrics:
                """Calculate comprehensive complexity metrics for a function"""
                metrics = ComplexityMetrics()
                
                # Calculate cyclomatic complexity
                decision_points = 0
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        decision_points += 1
                    elif isinstance(child, ast.BoolOp):
                        decision_points += len(child.values) - 1
                
                metrics.cyclomatic_complexity = decision_points + 1
                
                # Calculate cognitive complexity (simplified)
                nesting_level = 0
                cognitive_score = 0
                
                def calculate_cognitive(node, depth=0):
                    nonlocal cognitive_score
                    if isinstance(node, (ast.If, ast.While, ast.For)):
                        cognitive_score += depth + 1
                    elif isinstance(node, ast.Try):
                        cognitive_score += depth + 1
                    
                    for child in ast.iter_child_nodes(node):
                        new_depth = depth + 1 if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)) else depth
                        calculate_cognitive(child, new_depth)
                
                calculate_cognitive(node)
                metrics.cognitive_complexity = cognitive_score
                
                # Other metrics
                metrics.line_count = (node.end_lineno or node.lineno) - node.lineno + 1
                metrics.parameter_count = len(node.args.args)
                metrics.nesting_depth = self._calculate_max_nesting(node)
                
                return metrics
            
            def _calculate_max_nesting(self, node: ast.AST) -> int:
                """Calculate maximum nesting depth"""
                def get_depth(node, current_depth=0):
                    max_depth = current_depth
                    for child in ast.iter_child_nodes(node):
                        child_depth = current_depth
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                            child_depth += 1
                        max_depth = max(max_depth, get_depth(child, child_depth))
                    return max_depth
                
                return get_depth(node)
        
        analyzer = ComplexityAnalyzer()
        analyzer.thresholds = self.complexity_thresholds
        analyzer.visit(tree)
        
        return issues
    
    async def _analyze_maintainability(self, code: str, tree: ast.AST) -> List[QualityIssue]:
        """Analyze code maintainability factors"""
        issues = []
        
        # Magic number detection
        magic_number_pattern = r'(?<![a-zA-Z_])\d{2,}(?![a-zA-Z_])'
        magic_numbers = re.finditer(magic_number_pattern, code)
        
        for match in magic_numbers:
            number = match.group()
            if number not in ['100', '1000', '10']:  # Common acceptable numbers
                line_num = code[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    issue_type="magic_number",
                    severity="low",
                    confidence=0.6,
                    description=f"Magic number '{number}' should be replaced with named constant",
                    location=(line_num, line_num),
                    affected_element="literal",
                    code_snippet=number,
                    improvement_suggestion=f"Replace with named constant: SOME_CONSTANT = {number}",
                    maintainability_impact=0.1,
                    readability_impact=0.2,
                    technical_debt_hours=0.25
                ))
        
        # Long parameter list detection
        class MaintainabilityVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Dead code detection (unreachable code after return)
                has_return = False
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Return):
                        has_return = True
                        if i < len(node.body) - 1:  # More statements after return
                            issues.append(QualityIssue(
                                issue_type="unreachable_code",
                                severity="medium",
                                confidence=0.9,
                                description=f"Unreachable code after return in function '{node.name}'",
                                location=(node.body[i+1].lineno, node.end_lineno or node.lineno),
                                affected_element=node.name,
                                code_snippet="# Code after return statement",
                                improvement_suggestion="Remove unreachable code or restructure logic",
                                maintainability_impact=0.3,
                                readability_impact=0.4,
                                technical_debt_hours=0.5
                            ))
                
                self.generic_visit(node)
        
        visitor = MaintainabilityVisitor()
        visitor.visit(tree)
        
        return issues
    
    async def _analyze_readability(self, code: str, tree: ast.AST) -> List[QualityIssue]:
        """Analyze code readability factors"""
        issues = []
        
        class ReadabilityVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Single letter variable names (except common ones)
                acceptable_single_letters = {'i', 'j', 'k', 'x', 'y', 'z', '_'}
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and len(child.id) == 1:
                        if child.id not in acceptable_single_letters:
                            issues.append(QualityIssue(
                                issue_type="single_letter_variable",
                                severity="low",
                                confidence=0.5,
                                description=f"Single letter variable '{child.id}' reduces readability",
                                location=(getattr(child, 'lineno', 0), getattr(child, 'lineno', 0)),
                                affected_element=child.id,
                                code_snippet=child.id,
                                improvement_suggestion=f"Use descriptive name instead of '{child.id}'",
                                readability_impact=0.2,
                                maintainability_impact=0.1,
                                technical_debt_hours=0.1
                            ))
                
                # Nested ternary operators
                for child in ast.walk(node):
                    if isinstance(child, ast.IfExp):
                        # Check if body or orelse contains another IfExp
                        if any(isinstance(grandchild, ast.IfExp) for grandchild in ast.walk(child)):
                            issues.append(QualityIssue(
                                issue_type="nested_ternary",
                                severity="medium",
                                confidence=0.8,
                                description="Nested ternary operators reduce readability",
                                location=(getattr(child, 'lineno', 0), getattr(child, 'lineno', 0)),
                                affected_element="ternary operator",
                                code_snippet="nested ? expressions",
                                improvement_suggestion="Replace with explicit if-else statements",
                                readability_impact=0.4,
                                maintainability_impact=0.2,
                                technical_debt_hours=0.5
                            ))
                
                self.generic_visit(node)
            
            def visit_Name(self, node):
                # Abbreviation detection
                common_abbreviations = {'str', 'int', 'bool', 'dict', 'list', 'obj', 'val', 'var', 'tmp', 'temp'}
                if len(node.id) > 1 and node.id.lower() in common_abbreviations:
                    issues.append(QualityIssue(
                        issue_type="abbreviation_usage",
                        severity="low",
                        confidence=0.4,
                        description=f"Abbreviation '{node.id}' could be more descriptive",
                        location=(getattr(node, 'lineno', 0), getattr(node, 'lineno', 0)),
                        affected_element=node.id,
                        code_snippet=node.id,
                        improvement_suggestion=f"Consider more descriptive name than '{node.id}'",
                        readability_impact=0.1,
                        technical_debt_hours=0.1
                    ))
                
                self.generic_visit(node)
        
        visitor = ReadabilityVisitor()
        visitor.visit(tree)
        
        # Long line detection
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line) > 100:  # PEP 8 recommends 79, but being more lenient
                issues.append(QualityIssue(
                    issue_type="long_line",
                    severity="low",
                    confidence=0.6,
                    description=f"Line {i} exceeds recommended length ({len(line)} characters)",
                    location=(i, i),
                    affected_element="line",
                    code_snippet=line[:50] + "..." if len(line) > 50 else line,
                    improvement_suggestion="Break long line into multiple lines",
                    readability_impact=0.1,
                    technical_debt_hours=0.1
                ))
        
        return issues
    
    async def _analyze_documentation(self, code: str, tree: ast.AST) -> List[QualityIssue]:
        """Analyze documentation coverage and quality"""
        issues = []
        
        class DocumentationVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check for missing docstrings
                docstring = ast.get_docstring(node)
                if not docstring:
                    # Skip private methods and simple getters/setters
                    if not node.name.startswith('_') and len(node.body) > 1:
                        issues.append(QualityIssue(
                            issue_type="missing_function_docstring",
                            severity="medium",
                            confidence=0.8,
                            description=f"Function '{node.name}' is missing a docstring",
                            location=(node.lineno, node.lineno),
                            affected_element=node.name,
                            code_snippet=f"def {node.name}(...):",
                            improvement_suggestion="Add comprehensive docstring with description, parameters, and return value",
                            maintainability_impact=0.2,
                            readability_impact=0.3,
                            technical_debt_hours=0.5
                        ))
                elif len(docstring.split()) < 5:
                    issues.append(QualityIssue(
                        issue_type="inadequate_function_docstring",
                        severity="low",
                        confidence=0.6,
                        description=f"Function '{node.name}' has inadequate docstring",
                        location=(node.lineno, node.lineno),
                        affected_element=node.name,
                        code_snippet=f'"""{docstring}"""',
                        improvement_suggestion="Expand docstring with detailed description and parameter information",
                        maintainability_impact=0.1,
                        readability_impact=0.2,
                        technical_debt_hours=0.3
                    ))
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Check for missing class docstrings
                docstring = ast.get_docstring(node)
                if not docstring:
                    issues.append(QualityIssue(
                        issue_type="missing_class_docstring",
                        severity="medium",
                        confidence=0.9,
                        description=f"Class '{node.name}' is missing a docstring",
                        location=(node.lineno, node.lineno),
                        affected_element=node.name,
                        code_snippet=f"class {node.name}:",
                        improvement_suggestion="Add class docstring explaining purpose and usage",
                        maintainability_impact=0.2,
                        readability_impact=0.3,
                        technical_debt_hours=0.5
                    ))
                
                self.generic_visit(node)
        
        visitor = DocumentationVisitor()
        visitor.visit(tree)
        
        return issues
    
    async def _analyze_design_patterns(self, code: str, tree: ast.AST) -> List[QualityIssue]:
        """Analyze design patterns and anti-patterns"""
        issues = []
        
        class DesignPatternVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                # Singleton anti-pattern detection
                has_new = any(method.name == '__new__' for method in node.body if isinstance(method, ast.FunctionDef))
                has_instance_check = 'instance' in code.lower() and 'none' in code.lower()
                
                if has_new and has_instance_check:
                    issues.append(QualityIssue(
                        issue_type="singleton_antipattern",
                        severity="medium",
                        confidence=0.7,
                        description=f"Class '{node.name}' implements singleton pattern, which may hinder testing",
                        location=(node.lineno, node.end_lineno or node.lineno),
                        affected_element=node.name,
                        code_snippet=f"class {node.name}: # singleton implementation",
                        improvement_suggestion="Consider dependency injection or factory pattern instead",
                        maintainability_impact=0.3,
                        testability_impact=0.4,
                        technical_debt_hours=2.0
                    ))
                
                # Missing factory method opportunity
                if len([m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == '__init__']) == 1:
                    init_method = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == '__init__')
                    if len(init_method.args.args) > 4:  # Many constructor parameters
                        issues.append(QualityIssue(
                            issue_type="factory_method_opportunity",
                            severity="low",
                            confidence=0.5,
                            description=f"Class '{node.name}' constructor has many parameters, consider factory method",
                            location=(init_method.lineno, init_method.end_lineno or init_method.lineno),
                            affected_element=f"{node.name}.__init__",
                            code_snippet=f"def __init__(self, ...): # {len(init_method.args.args)} parameters",
                            improvement_suggestion="Consider implementing factory methods for different construction scenarios",
                            maintainability_impact=0.2,
                            testability_impact=0.2,
                            technical_debt_hours=1.0
                        ))
                
                self.generic_visit(node)
        
        visitor = DesignPatternVisitor()
        visitor.visit(tree)
        
        return issues
    
    async def _analyze_code_duplication(self, code: str, tree: ast.AST) -> List[QualityIssue]:
        """Analyze code duplication and suggest refactoring"""
        issues = []
        
        # Simple duplication detection using string matching
        lines = code.split('\n')
        stripped_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        line_counts = Counter(stripped_lines)
        duplicated_lines = {line: count for line, count in line_counts.items() if count > 2 and len(line) > 10}
        
        for duplicated_line, count in duplicated_lines.items():
            issues.append(QualityIssue(
                issue_type="code_duplication",
                severity="medium",
                confidence=0.6,
                description=f"Code line appears {count} times (potential duplication)",
                location=(0, 0),  # Would need more sophisticated analysis for exact locations
                affected_element="multiple locations",
                code_snippet=duplicated_line[:100] + "..." if len(duplicated_line) > 100 else duplicated_line,
                improvement_suggestion="Extract duplicated code into a reusable function or method",
                maintainability_impact=0.3,
                technical_debt_hours=count * 0.5
            ))
        
        return issues
    
    def _create_complexity_recommendations(self, issues: List[QualityIssue], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for complexity issues"""
        recommendations = []
        
        for issue in issues:
            if issue.issue_type in ["high_cyclomatic_complexity", "high_cognitive_complexity", "long_function", "too_many_parameters", "god_class"]:
                priority = OptimizationPriority.HIGH if issue.severity == "high" else OptimizationPriority.MEDIUM
                
                rec = create_optimization_recommendation(
                    OptimizationType.MAINTAINABILITY,
                    f"Complexity Reduction: {issue.issue_type.replace('_', ' ').title()}",
                    issue.description,
                    file_path,
                    priority,
                    OptimizationStrategy.REFACTOR
                )
                
                rec.target_lines = issue.location
                rec.target_element = issue.affected_element
                rec.original_code = issue.code_snippet
                rec.optimized_code = issue.improvement_suggestion
                rec.confidence_score = issue.confidence
                rec.estimated_hours = issue.technical_debt_hours
                rec.expected_improvement = {
                    'maintainability': issue.maintainability_impact * 100,
                    'testability': issue.testability_impact * 100,
                    'readability': issue.readability_impact * 100
                }
                
                rec.risk_assessment = create_risk_assessment(
                    implementation_risk=0.3,
                    technical_debt_risk=0.6,
                    business_impact_risk=0.2
                )
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_maintainability_recommendations(self, issues: List[QualityIssue], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for maintainability issues"""
        recommendations = []
        
        for issue in issues:
            if issue.issue_type in ["magic_number", "unreachable_code"]:
                rec = create_optimization_recommendation(
                    OptimizationType.MAINTAINABILITY,
                    f"Maintainability: {issue.issue_type.replace('_', ' ').title()}",
                    issue.description,
                    file_path,
                    OptimizationPriority.MEDIUM if issue.severity == "medium" else OptimizationPriority.LOW,
                    OptimizationStrategy.INCREMENTAL_IMPROVEMENT
                )
                
                rec.target_lines = issue.location
                rec.optimized_code = issue.improvement_suggestion
                rec.confidence_score = issue.confidence
                rec.estimated_hours = issue.technical_debt_hours
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_readability_recommendations(self, issues: List[QualityIssue], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for readability issues"""
        recommendations = []
        
        for issue in issues:
            if issue.issue_type in ["single_letter_variable", "nested_ternary", "abbreviation_usage", "long_line"]:
                rec = create_optimization_recommendation(
                    OptimizationType.READABILITY,
                    f"Readability: {issue.issue_type.replace('_', ' ').title()}",
                    issue.description,
                    file_path,
                    OptimizationPriority.MEDIUM if issue.severity == "medium" else OptimizationPriority.LOW,
                    OptimizationStrategy.GRADUAL
                )
                
                rec.target_lines = issue.location
                rec.original_code = issue.code_snippet
                rec.optimized_code = issue.improvement_suggestion
                rec.confidence_score = issue.confidence
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_documentation_recommendations(self, issues: List[QualityIssue], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for documentation issues"""
        recommendations = []
        
        for issue in issues:
            if issue.issue_type in ["missing_function_docstring", "missing_class_docstring", "inadequate_function_docstring"]:
                rec = create_optimization_recommendation(
                    OptimizationType.MAINTAINABILITY,
                    f"Documentation: {issue.issue_type.replace('_', ' ').title()}",
                    issue.description,
                    file_path,
                    OptimizationPriority.MEDIUM,
                    OptimizationStrategy.INCREMENTAL_IMPROVEMENT
                )
                
                rec.target_lines = issue.location
                rec.target_element = issue.affected_element
                rec.original_code = issue.code_snippet
                rec.optimized_code = issue.improvement_suggestion
                rec.confidence_score = issue.confidence
                rec.estimated_hours = issue.technical_debt_hours
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_design_recommendations(self, issues: List[QualityIssue], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for design pattern issues"""
        recommendations = []
        
        for issue in issues:
            if issue.issue_type in ["singleton_antipattern", "factory_method_opportunity"]:
                rec = create_optimization_recommendation(
                    OptimizationType.DESIGN_PATTERN,
                    f"Design Pattern: {issue.issue_type.replace('_', ' ').title()}",
                    issue.description,
                    file_path,
                    OptimizationPriority.MEDIUM,
                    OptimizationStrategy.REDESIGN
                )
                
                rec.target_lines = issue.location
                rec.target_element = issue.affected_element
                rec.optimized_code = issue.improvement_suggestion
                rec.confidence_score = issue.confidence
                rec.estimated_hours = issue.technical_debt_hours
                
                recommendations.append(rec)
        
        return recommendations
    
    def _create_duplication_recommendations(self, issues: List[QualityIssue], file_path: str) -> List[OptimizationRecommendation]:
        """Create recommendations for code duplication issues"""
        recommendations = []
        
        for issue in issues:
            if issue.issue_type == "code_duplication":
                rec = create_optimization_recommendation(
                    OptimizationType.REFACTORING,
                    "Code Duplication: Extract Common Functionality",
                    issue.description,
                    file_path,
                    OptimizationPriority.MEDIUM,
                    OptimizationStrategy.REFACTOR
                )
                
                rec.original_code = issue.code_snippet
                rec.optimized_code = issue.improvement_suggestion
                rec.confidence_score = issue.confidence
                rec.estimated_hours = issue.technical_debt_hours
                
                recommendations.append(rec)
        
        return recommendations
    
    def _prioritize_quality_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Apply quality-specific prioritization"""
        def quality_score(rec: OptimizationRecommendation) -> float:
            base_score = rec.confidence_score
            
            # Boost maintainability improvements
            if rec.optimization_type == OptimizationType.MAINTAINABILITY:
                base_score *= 1.2
            
            # Boost high technical debt items
            if rec.estimated_hours > 2.0:
                base_score *= 1.1
            
            # Boost items with high expected improvement
            total_improvement = sum(rec.expected_improvement.values())
            if total_improvement > 100:
                base_score *= 1.15
            
            return base_score
        
        return sorted(recommendations, key=quality_score, reverse=True)
    
    def _load_quality_patterns(self) -> Dict[str, Any]:
        """Load quality analysis patterns"""
        return {
            'complexity_patterns': [
                'high_cyclomatic', 'high_cognitive', 'deep_nesting',
                'long_methods', 'god_classes', 'feature_envy'
            ],
            'maintainability_patterns': [
                'magic_numbers', 'dead_code', 'duplicate_code',
                'long_parameter_lists', 'inappropriate_intimacy'
            ],
            'readability_patterns': [
                'unclear_naming', 'complex_expressions', 'poor_formatting',
                'missing_abstractions', 'inconsistent_style'
            ]
        }
    
    def _load_design_patterns(self) -> Dict[str, DesignPattern]:
        """Load design pattern database"""
        return {
            'factory_method': DesignPattern(
                pattern_name="Factory Method",
                pattern_type="creational",
                confidence=0.8,
                description="Create objects without specifying exact classes",
                benefits=["Loose coupling", "Easy testing", "Extensibility"],
                implementation_guidance="Replace complex constructors with factory methods"
            ),
            'strategy': DesignPattern(
                pattern_name="Strategy",
                pattern_type="behavioral", 
                confidence=0.7,
                description="Encapsulate algorithms and make them interchangeable",
                benefits=["Algorithm flexibility", "Easy testing", "Open/closed principle"],
                implementation_guidance="Extract conditional logic into strategy objects"
            )
        }
    
    def calculate_maintainability_index(self, metrics: ComplexityMetrics) -> float:
        """Calculate Maintainability Index using industry standard formula"""
        # Simplified version of the Microsoft Maintainability Index
        volume = metrics.line_count * math.log2(max(1, metrics.cyclomatic_complexity))
        complexity_factor = metrics.cyclomatic_complexity
        lines_factor = metrics.line_count
        
        # Normalize to 0-100 scale
        mi = max(0, min(100, 171 - 5.2 * math.log(volume) - 0.23 * complexity_factor - 16.2 * math.log(lines_factor)))
        return round(mi, 2)


# Factory function
def create_quality_analyzer() -> QualityAnalyzer:
    """Create and configure quality analyzer"""
    return QualityAnalyzer()