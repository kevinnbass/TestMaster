"""
Code Quality Debt Analyzer Component
====================================

Analyzes code-related technical debt including complexity,
duplication, and maintainability issues.
Part of modularized debt_analyzer system.
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from collections import defaultdict

from .debt_base import (
    DebtItem, DebtCategory, DebtSeverity,
    DebtConfiguration
)


class CodeDebtAnalyzer:
    """Analyzes code quality debt in Python files."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize code debt analyzer."""
        self.config = config or {}
        self.debt_config = DebtConfiguration()
        self.debt_items = []
        
        # Thresholds
        self.max_complexity = self.config.get('max_complexity', 10)
        self.max_function_lines = self.config.get('max_function_lines', 50)
        self.max_file_lines = self.config.get('max_file_lines', 300)
        self.min_name_length = self.config.get('min_name_length', 3)
    
    def analyze_code_debt(self, file_paths: List[Path]) -> List[DebtItem]:
        """Analyze code quality debt in given files."""
        self.debt_items = []
        
        for file_path in file_paths:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    self._analyze_complexity(tree, file_path)
                    self._analyze_code_duplication(tree, file_path)
                    self._analyze_naming_conventions(tree, file_path)
                    self._analyze_code_structure(tree, file_path)
                    self._analyze_dead_code(tree, file_path)
            except Exception:
                pass  # Skip files that can't be analyzed
        
        return self.debt_items
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse Python file into AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ast.parse(f.read())
        except:
            return None
    
    def _analyze_complexity(self, tree: ast.AST, file_path: Path):
        """Analyze cyclomatic complexity."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)
                
                if complexity > self.max_complexity:
                    severity = self._get_complexity_severity(complexity)
                    hours = self._estimate_refactoring_hours(complexity)
                    
                    self.debt_items.append(DebtItem(
                        type="complex_function",
                        severity=severity,
                        location=f"{file_path}:{node.lineno}:{node.name}",
                        description=f"Function '{node.name}' has complexity {complexity}",
                        estimated_hours=hours,
                        interest_rate=0.08,
                        risk_factor=1.5,
                        remediation_strategy="Refactor into smaller functions",
                        dependencies=[],
                        business_impact="Reduced maintainability and bug risk",
                        category=DebtCategory.CODE_QUALITY
                    ))
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += sum(1 for _ in child.ifs)
        
        return complexity
    
    def _get_complexity_severity(self, complexity: int) -> str:
        """Determine severity based on complexity."""
        if complexity > 30:
            return DebtSeverity.CRITICAL.value
        elif complexity > 20:
            return DebtSeverity.HIGH.value
        elif complexity > 15:
            return DebtSeverity.MEDIUM.value
        else:
            return DebtSeverity.LOW.value
    
    def _estimate_refactoring_hours(self, complexity: int) -> float:
        """Estimate hours needed to refactor based on complexity."""
        base_hours = self.debt_config.DEBT_COST_FACTORS["complex_function"]
        complexity_factor = (complexity - self.max_complexity) / 10
        return base_hours * (1 + complexity_factor)
    
    def _analyze_code_duplication(self, tree: ast.AST, file_path: Path):
        """Detect code duplication patterns."""
        # Simplified duplication detection
        function_bodies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body_str = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                
                # Check for similar functions
                for other_body in function_bodies:
                    if self._is_similar(body_str, other_body):
                        self.debt_items.append(DebtItem(
                            type="code_duplication",
                            severity=DebtSeverity.MEDIUM.value,
                            location=str(file_path),
                            description="Duplicated code pattern detected",
                            estimated_hours=self.debt_config.DEBT_COST_FACTORS["code_duplication"],
                            interest_rate=0.06,
                            risk_factor=1.2,
                            remediation_strategy="Extract common functionality",
                            dependencies=[],
                            business_impact="Increased maintenance effort",
                            category=DebtCategory.CODE_QUALITY
                        ))
                        break
                
                function_bodies.append(body_str)
    
    def _is_similar(self, code1: str, code2: str, threshold: float = 0.8) -> bool:
        """Check if two code snippets are similar."""
        # Simplified similarity check
        if len(code1) < 50 or len(code2) < 50:
            return False
        
        # Basic token comparison
        tokens1 = set(re.findall(r'\w+', code1))
        tokens2 = set(re.findall(r'\w+', code2))
        
        if not tokens1 or not tokens2:
            return False
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def _analyze_naming_conventions(self, tree: ast.AST, file_path: Path):
        """Check for poor naming conventions."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if len(node.name) < self.min_name_length or node.name.lower() in ['func', 'fn', 'tmp', 'test']:
                    self.debt_items.append(DebtItem(
                        type="poor_naming",
                        severity=DebtSeverity.LOW.value,
                        location=f"{file_path}:{node.lineno}",
                        description=f"Poor naming: '{node.name}'",
                        estimated_hours=self.debt_config.DEBT_COST_FACTORS["poor_naming"],
                        interest_rate=0.04,
                        risk_factor=1.0,
                        remediation_strategy="Use descriptive names",
                        dependencies=[],
                        business_impact="Reduced code readability",
                        category=DebtCategory.CODE_QUALITY
                    ))
    
    def _analyze_code_structure(self, tree: ast.AST, file_path: Path):
        """Analyze overall code structure issues."""
        # Check file length
        try:
            with open(file_path, 'r') as f:
                line_count = len(f.readlines())
            
            if line_count > self.max_file_lines:
                self.debt_items.append(DebtItem(
                    type="large_file",
                    severity=DebtSeverity.MEDIUM.value,
                    location=str(file_path),
                    description=f"File has {line_count} lines (max: {self.max_file_lines})",
                    estimated_hours=4.0,
                    interest_rate=0.06,
                    risk_factor=1.3,
                    remediation_strategy="Split into smaller modules",
                    dependencies=[],
                    business_impact="Difficult to navigate and maintain",
                    category=DebtCategory.CODE_QUALITY
                ))
        except:
            pass
    
    def _analyze_dead_code(self, tree: ast.AST, file_path: Path):
        """Detect potentially dead code."""
        # Check for unreachable code after return/raise
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._check_unreachable_code(node, file_path)
    
    def _check_unreachable_code(self, func_node: ast.AST, file_path: Path):
        """Check for unreachable code in function."""
        found_return = False
        
        for stmt in func_node.body:
            if found_return:
                self.debt_items.append(DebtItem(
                    type="dead_code",
                    severity=DebtSeverity.LOW.value,
                    location=f"{file_path}:{stmt.lineno}",
                    description="Unreachable code detected",
                    estimated_hours=self.debt_config.DEBT_COST_FACTORS["dead_code"],
                    interest_rate=0.02,
                    risk_factor=1.0,
                    remediation_strategy="Remove dead code",
                    dependencies=[],
                    business_impact="Code confusion and maintenance overhead",
                    category=DebtCategory.CODE_QUALITY
                ))
                break
            
            if isinstance(stmt, (ast.Return, ast.Raise)):
                found_return = True


# Export
__all__ = ['CodeDebtAnalyzer']