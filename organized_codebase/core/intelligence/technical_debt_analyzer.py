"""
Technical Debt Analysis Module
Quantifies technical debt in developer-hours and provides remediation strategies
Extracted from archive and integrated into core intelligence system
"""

import ast
import re
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import json
import math


@dataclass
class DebtItem:
    """Represents a single technical debt item"""
    type: str
    severity: str
    location: str
    description: str
    estimated_hours: float
    interest_rate: float  # How much this debt grows over time
    risk_factor: float
    remediation_strategy: str
    dependencies: List[str] = field(default_factory=list)
    business_impact: str = "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'type': self.type,
            'severity': self.severity,
            'location': self.location,
            'description': self.description,
            'estimated_hours': self.estimated_hours,
            'interest_rate': self.interest_rate,
            'risk_factor': self.risk_factor,
            'remediation_strategy': self.remediation_strategy,
            'dependencies': self.dependencies,
            'business_impact': self.business_impact
        }


@dataclass
class DebtMetrics:
    """Aggregate technical debt metrics"""
    total_debt_hours: float
    debt_ratio: float  # Debt hours / total development hours
    monthly_interest: float  # Additional hours added per month
    break_even_point: int  # Months until fixing debt pays off
    risk_adjusted_cost: float
    team_velocity_impact: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'total_debt_hours': self.total_debt_hours,
            'debt_ratio': self.debt_ratio,
            'monthly_interest': self.monthly_interest,
            'break_even_point': self.break_even_point,
            'risk_adjusted_cost': self.risk_adjusted_cost,
            'team_velocity_impact': self.team_velocity_impact
        }


class TechnicalDebtAnalyzer:
    """
    Analyzes and quantifies technical debt in developer-hours
    Provides actionable remediation strategies and prioritization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Cost factors (hours to fix)
        self.debt_cost_factors = {
            "code_duplication": 2.0,  # Per duplication instance
            "missing_tests": 1.5,  # Per untested function
            "complex_function": 4.0,  # Per high-complexity function
            "poor_naming": 0.5,  # Per poorly named element
            "missing_documentation": 1.0,  # Per undocumented public API
            "deprecated_usage": 2.5,  # Per deprecated API usage
            "security_vulnerability": 8.0,  # Per security issue
            "performance_issue": 6.0,  # Per performance bottleneck
            "architectural_violation": 12.0,  # Per major architectural issue
            "dead_code": 0.5,  # Per dead code block
            "inconsistent_style": 0.25,  # Per style violation
            "tight_coupling": 5.0,  # Per tightly coupled component
            "missing_error_handling": 3.0,  # Per unhandled error path
            "hardcoded_values": 1.0,  # Per hardcoded configuration
            "outdated_dependency": 4.0  # Per outdated dependency
        }
        
        # Interest rates (monthly growth factor)
        self.interest_rates = {
            "security": 0.15,  # 15% monthly growth
            "performance": 0.10,  # 10% monthly growth
            "maintainability": 0.08,  # 8% monthly growth
            "reliability": 0.12,  # 12% monthly growth
            "testability": 0.06,  # 6% monthly growth
            "documentation": 0.04,  # 4% monthly growth
            "code_quality": 0.05  # 5% monthly growth
        }
        
        # Team productivity factors
        self.productivity_factors = {
            "junior_developer": 1.5,  # Takes 1.5x longer
            "mid_developer": 1.0,  # Baseline
            "senior_developer": 0.7,  # 30% faster
            "team_size_factor": 1.2,  # Coordination overhead
            "context_switching": 1.3  # Cost of switching between tasks
        }
        
        self.debt_items = []
        self.debt_metrics = None
        
    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive technical debt analysis on a project
        """
        try:
            project_path = Path(project_path)
            self.debt_items = []
            
            # Analyze Python files
            for py_file in project_path.rglob("*.py"):
                if self._should_analyze_file(py_file):
                    self._analyze_file(py_file)
            
            # Calculate metrics
            self.debt_metrics = self._calculate_debt_metrics()
            
            return {
                'debt_items': [item.to_dict() for item in self.debt_items],
                'metrics': self.debt_metrics.to_dict() if self.debt_metrics else None,
                'summary': self._generate_summary(),
                'recommendations': self._generate_recommendations(),
                'prioritized_actions': self._prioritize_debt_items(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'debt_items': [],
                'metrics': None,
                'timestamp': datetime.now().isoformat()
            }
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed"""
        # Skip only essential excluded directories, include test files for TestMaster analysis
        skip_patterns = [
            '__pycache__/', '.git/', 'node_modules/', 'venv/', '.venv/',
            '.pytest_cache/', 'migrations/'
        ]
        
        file_str = str(file_path)
        # Skip only if file is in excluded directories, not if it contains test patterns
        for pattern in skip_patterns:
            if pattern in file_str:
                return False
        
        # Skip empty files
        try:
            if file_path.stat().st_size == 0:
                return False
        except OSError:
            return False
            
        return True
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file for technical debt"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze different types of debt
            self._detect_code_duplication(file_path, content)
            self._detect_missing_tests(file_path, tree)
            self._detect_complex_functions(file_path, tree)
            self._detect_poor_naming(file_path, tree)
            self._detect_missing_documentation(file_path, tree)
            self._detect_dead_code(file_path, tree)
            self._detect_security_issues(file_path, content)
            self._detect_performance_issues(file_path, tree, content)
            self._detect_architectural_violations(file_path, tree)
            
        except Exception as e:
            # Log error but continue analysis
            pass
    
    def _detect_code_duplication(self, file_path: Path, content: str):
        """Detect code duplication patterns"""
        lines = content.split('\n')
        duplicates = 0
        
        # Simple duplicate detection (could be enhanced)
        line_counts = defaultdict(int)
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 20 and not stripped.startswith('#'):
                line_counts[stripped] += 1
        
        for line, count in line_counts.items():
            if count > 2:  # Appears 3+ times
                duplicates += count - 1
        
        if duplicates > 0:
            debt_item = DebtItem(
                type="code_duplication",
                severity="medium",
                location=str(file_path),
                description=f"Found {duplicates} duplicate code patterns",
                estimated_hours=duplicates * self.debt_cost_factors["code_duplication"],
                interest_rate=self.interest_rates["maintainability"],
                risk_factor=0.6,
                remediation_strategy="Extract common functionality into shared functions/classes"
            )
            self.debt_items.append(debt_item)
    
    def _detect_missing_tests(self, file_path: Path, tree: ast.AST):
        """Detect functions/classes without test coverage"""
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):  # Public functions
                    functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        # Assume no test coverage for now (would integrate with coverage tools)
        untested_count = len(functions) + len(classes)
        
        if untested_count > 0:
            debt_item = DebtItem(
                type="missing_tests",
                severity="high",
                location=str(file_path),
                description=f"Found {untested_count} untested functions/classes",
                estimated_hours=untested_count * self.debt_cost_factors["missing_tests"],
                interest_rate=self.interest_rates["testability"],
                risk_factor=0.8,
                remediation_strategy="Write comprehensive unit tests for public APIs"
            )
            self.debt_items.append(debt_item)
    
    def _detect_complex_functions(self, file_path: Path, tree: ast.AST):
        """Detect functions with high cyclomatic complexity"""
        complex_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                if complexity > 10:  # Threshold for high complexity
                    complex_functions += 1
        
        if complex_functions > 0:
            debt_item = DebtItem(
                type="complex_function",
                severity="high",
                location=str(file_path),
                description=f"Found {complex_functions} overly complex functions",
                estimated_hours=complex_functions * self.debt_cost_factors["complex_function"],
                interest_rate=self.interest_rates["maintainability"],
                risk_factor=0.7,
                remediation_strategy="Refactor complex functions into smaller, focused functions"
            )
            self.debt_items.append(debt_item)
    
    def _detect_poor_naming(self, file_path: Path, tree: ast.AST):
        """Detect poor naming conventions"""
        poor_names = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Name)):
                name = getattr(node, 'name', None) or getattr(node, 'id', None)
                if name and self._is_poor_name(name):
                    poor_names += 1
        
        if poor_names > 0:
            debt_item = DebtItem(
                type="poor_naming",
                severity="low",
                location=str(file_path),
                description=f"Found {poor_names} poorly named elements",
                estimated_hours=poor_names * self.debt_cost_factors["poor_naming"],
                interest_rate=self.interest_rates["maintainability"],
                risk_factor=0.3,
                remediation_strategy="Rename variables, functions, and classes to be more descriptive"
            )
            self.debt_items.append(debt_item)
    
    def _detect_missing_documentation(self, file_path: Path, tree: ast.AST):
        """Detect missing documentation"""
        undocumented_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith('_'):  # Public APIs
                    if not ast.get_docstring(node):
                        undocumented_count += 1
        
        if undocumented_count > 0:
            debt_item = DebtItem(
                type="missing_documentation",
                severity="medium",
                location=str(file_path),
                description=f"Found {undocumented_count} undocumented public APIs",
                estimated_hours=undocumented_count * self.debt_cost_factors["missing_documentation"],
                interest_rate=self.interest_rates["documentation"],
                risk_factor=0.4,
                remediation_strategy="Add comprehensive docstrings to all public APIs"
            )
            self.debt_items.append(debt_item)
    
    def _detect_dead_code(self, file_path: Path, tree: ast.AST):
        """Detect dead/unreachable code"""
        # Simplified dead code detection
        dead_code_blocks = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for always-false conditions
                if isinstance(node.test, ast.Constant) and not node.test.value:
                    dead_code_blocks += 1
        
        if dead_code_blocks > 0:
            debt_item = DebtItem(
                type="dead_code",
                severity="low",
                location=str(file_path),
                description=f"Found {dead_code_blocks} dead code blocks",
                estimated_hours=dead_code_blocks * self.debt_cost_factors["dead_code"],
                interest_rate=self.interest_rates["maintainability"],
                risk_factor=0.2,
                remediation_strategy="Remove dead/unreachable code"
            )
            self.debt_items.append(debt_item)
    
    def _detect_security_issues(self, file_path: Path, content: str):
        """Detect potential security issues"""
        security_issues = 0
        
        # Simple security pattern detection
        security_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.call\s*\(',
            r'password\s*=\s*["\'][^"\']*["\']',  # Hardcoded passwords
            r'secret\s*=\s*["\'][^"\']*["\']',    # Hardcoded secrets
        ]
        
        for pattern in security_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            security_issues += len(matches)
        
        if security_issues > 0:
            debt_item = DebtItem(
                type="security_vulnerability",
                severity="critical",
                location=str(file_path),
                description=f"Found {security_issues} potential security issues",
                estimated_hours=security_issues * self.debt_cost_factors["security_vulnerability"],
                interest_rate=self.interest_rates["security"],
                risk_factor=0.9,
                remediation_strategy="Fix security vulnerabilities and implement secure coding practices",
                business_impact="high"
            )
            self.debt_items.append(debt_item)
    
    def _detect_performance_issues(self, file_path: Path, tree: ast.AST, content: str):
        """Detect potential performance issues"""
        performance_issues = 0
        
        # Simple performance pattern detection
        for node in ast.walk(tree):
            # Nested loops
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        performance_issues += 1
                        break
        
        if performance_issues > 0:
            debt_item = DebtItem(
                type="performance_issue",
                severity="medium",
                location=str(file_path),
                description=f"Found {performance_issues} potential performance issues",
                estimated_hours=performance_issues * self.debt_cost_factors["performance_issue"],
                interest_rate=self.interest_rates["performance"],
                risk_factor=0.6,
                remediation_strategy="Optimize algorithms and data structures for better performance"
            )
            self.debt_items.append(debt_item)
    
    def _detect_architectural_violations(self, file_path: Path, tree: ast.AST):
        """Detect architectural violations"""
        violations = 0
        
        # Simple architectural checks
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check for problematic imports
                    if any(bad in alias.name for bad in ['sqlite3', 'mysql', 'postgres'] 
                          if 'models' not in str(file_path).lower()):
                        violations += 1
        
        if violations > 0:
            debt_item = DebtItem(
                type="architectural_violation",
                severity="high",
                location=str(file_path),
                description=f"Found {violations} architectural violations",
                estimated_hours=violations * self.debt_cost_factors["architectural_violation"],
                interest_rate=self.interest_rates["maintainability"],
                risk_factor=0.8,
                remediation_strategy="Refactor to follow proper architectural patterns and layering"
            )
            self.debt_items.append(debt_item)
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _is_poor_name(self, name: str) -> bool:
        """Check if a name follows poor naming conventions"""
        poor_patterns = [
            r'^[a-z]$',  # Single letter
            r'^\d+$',    # Just numbers
            r'^(data|info|obj|item|thing|stuff)$',  # Generic names
            r'^(a|an|the|my|your)_',  # Article prefixes
        ]
        
        return any(re.match(pattern, name) for pattern in poor_patterns)
    
    def _calculate_debt_metrics(self) -> DebtMetrics:
        """Calculate aggregate debt metrics"""
        if not self.debt_items:
            return DebtMetrics(0, 0, 0, 0, 0, 0)
        
        total_debt_hours = sum(item.estimated_hours for item in self.debt_items)
        
        # Estimate total development hours (simplified)
        total_dev_hours = len(self.debt_items) * 40  # Rough estimate
        debt_ratio = total_debt_hours / max(total_dev_hours, 1)
        
        # Calculate monthly interest
        monthly_interest = sum(
            item.estimated_hours * item.interest_rate 
            for item in self.debt_items
        )
        
        # Break-even calculation
        break_even_point = int(total_debt_hours / max(monthly_interest, 1)) if monthly_interest > 0 else 999
        
        # Risk-adjusted cost
        risk_adjusted_cost = sum(
            item.estimated_hours * item.risk_factor 
            for item in self.debt_items
        )
        
        # Team velocity impact (simplified)
        team_velocity_impact = min(debt_ratio * 0.3, 0.5)  # Max 50% impact
        
        return DebtMetrics(
            total_debt_hours=total_debt_hours,
            debt_ratio=debt_ratio,
            monthly_interest=monthly_interest,
            break_even_point=break_even_point,
            risk_adjusted_cost=risk_adjusted_cost,
            team_velocity_impact=team_velocity_impact
        )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of debt analysis"""
        if not self.debt_items:
            return {"message": "No technical debt detected"}
        
        debt_by_type = defaultdict(int)
        debt_by_severity = defaultdict(int)
        
        for item in self.debt_items:
            debt_by_type[item.type] += 1
            debt_by_severity[item.severity] += 1
        
        return {
            "total_items": len(self.debt_items),
            "debt_by_type": dict(debt_by_type),
            "debt_by_severity": dict(debt_by_severity),
            "top_debt_types": sorted(debt_by_type.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        if not self.debt_items:
            return ["Codebase appears to be in good health!"]
        
        recommendations = []
        
        # High-priority recommendations based on severity
        critical_items = [item for item in self.debt_items if item.severity == "critical"]
        high_items = [item for item in self.debt_items if item.severity == "high"]
        
        if critical_items:
            recommendations.append(f"URGENT: Address {len(critical_items)} critical issues immediately")
        
        if high_items:
            recommendations.append(f"HIGH PRIORITY: Plan to fix {len(high_items)} high-severity issues")
        
        # Type-specific recommendations
        debt_by_type = defaultdict(int)
        for item in self.debt_items:
            debt_by_type[item.type] += 1
        
        if debt_by_type.get("security_vulnerability", 0) > 0:
            recommendations.append("Implement security code review process")
        
        if debt_by_type.get("missing_tests", 0) > 5:
            recommendations.append("Establish test coverage targets and CI enforcement")
        
        if debt_by_type.get("complex_function", 0) > 3:
            recommendations.append("Refactor complex functions to improve maintainability")
        
        return recommendations
    
    def _prioritize_debt_items(self) -> List[Dict[str, Any]]:
        """Prioritize debt items by impact and effort"""
        prioritized = sorted(
            self.debt_items,
            key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.severity],
                x.risk_factor,
                -x.estimated_hours  # Prefer easier fixes for same severity
            ),
            reverse=True
        )
        
        return [
            {
                "rank": i + 1,
                "item": item.to_dict(),
                "priority_score": (
                    {"critical": 4, "high": 3, "medium": 2, "low": 1}[item.severity] * 
                    item.risk_factor
                )
            }
            for i, item in enumerate(prioritized[:20])  # Top 20 items
        ]


# Export
__all__ = ['TechnicalDebtAnalyzer', 'DebtItem', 'DebtMetrics']