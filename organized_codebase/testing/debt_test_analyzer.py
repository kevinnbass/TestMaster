"""
Test Debt Analyzer Component
============================

Analyzes testing-related technical debt including missing tests,
low coverage, and test quality issues.
Part of modularized debt_analyzer system.
"""

import ast
import os
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from collections import defaultdict

from .debt_base import (
    DebtItem, DebtCategory, DebtSeverity,
    DebtConfiguration
)


class TestDebtAnalyzer:
    """Analyzes test-related technical debt."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize test debt analyzer."""
        self.config = config or {}
        self.debt_config = DebtConfiguration()
        self.debt_items = []
        
        # Coverage thresholds
        self.min_coverage = self.config.get('min_coverage', 80.0)
        self.min_test_ratio = self.config.get('min_test_ratio', 0.5)  # tests per function
        
        # Test patterns
        self.test_patterns = ['test_', '_test', 'Test', 'spec_', '_spec']
    
    def analyze_test_debt(self, 
                         source_files: List[Path],
                         test_files: List[Path],
                         coverage_data: Optional[Dict[str, float]] = None) -> List[DebtItem]:
        """Analyze test-related debt."""
        self.debt_items = []
        
        # Analyze coverage
        if coverage_data:
            self._analyze_coverage_debt(coverage_data)
        
        # Analyze missing tests
        self._analyze_missing_tests(source_files, test_files)
        
        # Analyze test quality
        self._analyze_test_quality(test_files)
        
        # Analyze test organization
        self._analyze_test_organization(test_files)
        
        return self.debt_items
    
    def _analyze_coverage_debt(self, coverage_data: Dict[str, float]):
        """Analyze test coverage debt."""
        total_coverage = coverage_data.get('total_coverage', 0)
        
        if total_coverage < self.min_coverage:
            gap = self.min_coverage - total_coverage
            severity = self._get_coverage_severity(total_coverage)
            hours = self._estimate_test_writing_hours(gap)
            
            self.debt_items.append(DebtItem(
                type="low_coverage",
                severity=severity,
                location="project",
                description=f"Test coverage is {total_coverage:.1f}% (target: {self.min_coverage}%)",
                estimated_hours=hours,
                interest_rate=0.06,
                risk_factor=1.8,
                remediation_strategy="Write additional unit tests",
                dependencies=[],
                business_impact="Increased bug risk and regression potential",
                category=DebtCategory.TESTING
            ))
        
        # Check file-level coverage
        for file_path, coverage in coverage_data.items():
            if file_path != 'total_coverage' and coverage < self.min_coverage:
                self.debt_items.append(DebtItem(
                    type="missing_tests",
                    severity=DebtSeverity.MEDIUM.value,
                    location=str(file_path),
                    description=f"File has {coverage:.1f}% coverage",
                    estimated_hours=self.debt_config.DEBT_COST_FACTORS["missing_tests"],
                    interest_rate=0.05,
                    risk_factor=1.5,
                    remediation_strategy="Add unit tests for uncovered code",
                    dependencies=[],
                    business_impact="Untested code paths may contain bugs",
                    category=DebtCategory.TESTING
                ))
    
    def _get_coverage_severity(self, coverage: float) -> str:
        """Determine severity based on coverage percentage."""
        if coverage < 20:
            return DebtSeverity.CRITICAL.value
        elif coverage < 40:
            return DebtSeverity.HIGH.value
        elif coverage < 60:
            return DebtSeverity.MEDIUM.value
        else:
            return DebtSeverity.LOW.value
    
    def _estimate_test_writing_hours(self, coverage_gap: float) -> float:
        """Estimate hours needed to close coverage gap."""
        # Rough estimate: 1 hour per 5% coverage increase
        return (coverage_gap / 5) * self.debt_config.DEBT_COST_FACTORS["missing_tests"]
    
    def _analyze_missing_tests(self, source_files: List[Path], test_files: List[Path]):
        """Identify source files without corresponding tests."""
        # Build test file mapping
        tested_modules = set()
        for test_file in test_files:
            # Extract module names from test files
            content = self._read_file(test_file)
            if content:
                imports = self._extract_imports(content)
                tested_modules.update(imports)
        
        # Check each source file
        for source_file in source_files:
            module_name = source_file.stem
            if module_name not in tested_modules and not self._is_test_file(source_file):
                functions = self._count_functions(source_file)
                if functions > 0:
                    hours = functions * self.debt_config.DEBT_COST_FACTORS["missing_tests"]
                    
                    self.debt_items.append(DebtItem(
                        type="missing_tests",
                        severity=DebtSeverity.HIGH.value,
                        location=str(source_file),
                        description=f"No tests found for module with {functions} functions",
                        estimated_hours=hours,
                        interest_rate=0.07,
                        risk_factor=2.0,
                        remediation_strategy="Create comprehensive test suite",
                        dependencies=[],
                        business_impact="Completely untested functionality",
                        category=DebtCategory.TESTING
                    ))
    
    def _analyze_test_quality(self, test_files: List[Path]):
        """Analyze quality of existing tests."""
        for test_file in test_files:
            tree = self._parse_file(test_file)
            if tree:
                # Check for test anti-patterns
                self._check_test_assertions(tree, test_file)
                self._check_test_naming(tree, test_file)
                self._check_test_structure(tree, test_file)
    
    def _check_test_assertions(self, tree: ast.AST, test_file: Path):
        """Check for tests without assertions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                has_assertion = self._has_assertion(node)
                
                if not has_assertion:
                    self.debt_items.append(DebtItem(
                        type="test_without_assertion",
                        severity=DebtSeverity.MEDIUM.value,
                        location=f"{test_file}:{node.lineno}",
                        description=f"Test '{node.name}' has no assertions",
                        estimated_hours=0.5,
                        interest_rate=0.04,
                        risk_factor=1.3,
                        remediation_strategy="Add meaningful assertions",
                        dependencies=[],
                        business_impact="Test doesn't verify behavior",
                        category=DebtCategory.TESTING
                    ))
    
    def _has_assertion(self, node: ast.FunctionDef) -> bool:
        """Check if function contains assertions."""
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                return True
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if 'assert' in child.func.id.lower() or 'expect' in child.func.id.lower():
                        return True
                elif isinstance(child.func, ast.Attribute):
                    if 'assert' in child.func.attr.lower() or 'expect' in child.func.attr.lower():
                        return True
        return False
    
    def _check_test_naming(self, tree: ast.AST, test_file: Path):
        """Check for poorly named tests."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Check for generic names
                if node.name in ['test_', 'test_1', 'test_2', 'test_function', 'test_method']:
                    self.debt_items.append(DebtItem(
                        type="poor_test_naming",
                        severity=DebtSeverity.LOW.value,
                        location=f"{test_file}:{node.lineno}",
                        description=f"Test has generic name: '{node.name}'",
                        estimated_hours=0.25,
                        interest_rate=0.02,
                        risk_factor=1.0,
                        remediation_strategy="Use descriptive test names",
                        dependencies=[],
                        business_impact="Unclear test purpose",
                        category=DebtCategory.TESTING
                    ))
    
    def _check_test_structure(self, tree: ast.AST, test_file: Path):
        """Check for test structure issues."""
        test_functions = [n for n in ast.walk(tree) 
                         if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
        
        # Check for test file size
        if len(test_functions) > 20:
            self.debt_items.append(DebtItem(
                type="large_test_file",
                severity=DebtSeverity.LOW.value,
                location=str(test_file),
                description=f"Test file has {len(test_functions)} tests",
                estimated_hours=2.0,
                interest_rate=0.03,
                risk_factor=1.1,
                remediation_strategy="Split into focused test modules",
                dependencies=[],
                business_impact="Difficult to maintain and understand",
                category=DebtCategory.TESTING
            ))
    
    def _analyze_test_organization(self, test_files: List[Path]):
        """Analyze test organization and structure."""
        # Check for test directory structure
        test_dirs = set(f.parent for f in test_files)
        
        if len(test_dirs) == 1:
            # All tests in single directory
            if len(test_files) > 10:
                self.debt_items.append(DebtItem(
                    type="poor_test_organization",
                    severity=DebtSeverity.LOW.value,
                    location="tests/",
                    description="All tests in single directory",
                    estimated_hours=3.0,
                    interest_rate=0.03,
                    risk_factor=1.1,
                    remediation_strategy="Organize tests by module/feature",
                    dependencies=[],
                    business_impact="Hard to find relevant tests",
                    category=DebtCategory.TESTING
                ))
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return None
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse Python file into AST."""
        content = self._read_file(file_path)
        if content:
            try:
                return ast.parse(content)
            except:
                pass
        return None
    
    def _extract_imports(self, content: str) -> Set[str]:
        """Extract imported module names from content."""
        imports = set()
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except:
            pass
        return imports
    
    def _count_functions(self, file_path: Path) -> int:
        """Count number of functions in file."""
        tree = self._parse_file(file_path)
        if tree:
            return sum(1 for n in ast.walk(tree) 
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        return 0
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        name = file_path.stem
        return any(pattern in name for pattern in self.test_patterns)


# Export
__all__ = ['TestDebtAnalyzer']