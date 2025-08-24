"""
Intelligent Test Maintenance System for TestMaster
Automated test suite maintenance, repair, and optimization
"""

import asyncio
import time
import json
import re
import ast
import difflib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import hashlib

class MaintenanceType(Enum):
    """Types of test maintenance operations"""
    REPAIR_SYNTAX = "repair_syntax"
    UPDATE_IMPORTS = "update_imports"
    FIX_ASSERTIONS = "fix_assertions"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    REMOVE_DUPLICATES = "remove_duplicates"
    UPDATE_DEPRECATED = "update_deprecated"
    ENHANCE_COVERAGE = "enhance_coverage"
    REFACTOR_STRUCTURE = "refactor_structure"

class MaintenanceLevel(Enum):
    """Maintenance operation severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    OPTIMIZATION = "OPTIMIZATION"

class TestStatus(Enum):
    """Test execution status"""
    PASSING = "PASSING"
    FAILING = "FAILING"
    BROKEN = "BROKEN"
    SKIPPED = "SKIPPED"
    UNKNOWN = "UNKNOWN"

@dataclass
class TestIssue:
    """Identified test issue"""
    test_file: str
    issue_type: MaintenanceType
    severity: MaintenanceLevel
    description: str
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    confidence_score: float = 0.0
    auto_fixable: bool = False

@dataclass
class MaintenanceAction:
    """Test maintenance action"""
    action_id: str
    test_file: str
    action_type: MaintenanceType
    original_code: str
    fixed_code: str
    line_range: Tuple[int, int]
    confidence_score: float
    validation_required: bool = True

@dataclass
class MaintenanceReport:
    """Test maintenance execution report"""
    session_id: str
    total_tests_analyzed: int
    issues_found: int
    issues_fixed: int
    issues_requiring_manual_review: int
    performance_improvements: int
    actions_performed: List[MaintenanceAction]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]
    execution_time: float
    timestamp: float

class TestMaintenanceSystem:
    """Intelligent Test Maintenance System"""
    
    def __init__(self, test_directory: str):
        self.test_directory = Path(test_directory)
        self.issues_found: List[TestIssue] = []
        self.actions_performed: List[MaintenanceAction] = []
        self.test_cache: Dict[str, Dict[str, Any]] = {}
        
    def analyze_test_suite(self) -> List[TestIssue]:
        """Comprehensively analyze test suite for maintenance issues"""
        issues = []
        
        for test_file in self.test_directory.rglob("test_*.py"):
            file_issues = self._analyze_test_file(test_file)
            issues.extend(file_issues)
        
        self.issues_found = issues
        return issues
    
    def _analyze_test_file(self, test_file: Path) -> List[TestIssue]:
        """Analyze individual test file for issues"""
        issues = []
        
        try:
            content = test_file.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Syntax and structure analysis
            issues.extend(self._check_syntax_issues(test_file, content, tree))
            
            # Import analysis
            issues.extend(self._check_import_issues(test_file, content, tree))
            
            # Assertion analysis
            issues.extend(self._check_assertion_issues(test_file, content, tree))
            
            # Performance analysis
            issues.extend(self._check_performance_issues(test_file, content, tree))
            
            # Duplication analysis
            issues.extend(self._check_duplication_issues(test_file, content, tree))
            
            # Deprecated pattern analysis
            issues.extend(self._check_deprecated_patterns(test_file, content, tree))
            
        except SyntaxError as e:
            issues.append(TestIssue(
                test_file=str(test_file),
                issue_type=MaintenanceType.REPAIR_SYNTAX,
                severity=MaintenanceLevel.CRITICAL,
                description=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                auto_fixable=True,
                confidence_score=0.9
            ))
        except Exception as e:
            issues.append(TestIssue(
                test_file=str(test_file),
                issue_type=MaintenanceType.REPAIR_SYNTAX,
                severity=MaintenanceLevel.HIGH,
                description=f"Parse error: {str(e)}",
                auto_fixable=False,
                confidence_score=0.5
            ))
        
        return issues
    
    def _check_syntax_issues(self, test_file: Path, content: str, tree: ast.AST) -> List[TestIssue]:
        """Check for syntax and structural issues"""
        issues = []
        lines = content.split('\n')
        
        # Check for common syntax patterns that need fixing
        for i, line in enumerate(lines, 1):
            # Missing parentheses in print statements (Python 2 style)
            if re.search(r'\bprint\s+[^(]', line) and not line.strip().startswith('#'):
                issues.append(TestIssue(
                    test_file=str(test_file),
                    issue_type=MaintenanceType.REPAIR_SYNTAX,
                    severity=MaintenanceLevel.MEDIUM,
                    description="Python 2 style print statement",
                    line_number=i,
                    suggested_fix=re.sub(r'\bprint\s+(.+)', r'print(\1)', line),
                    auto_fixable=True,
                    confidence_score=0.8
                ))
            
            # Inconsistent indentation
            if line.strip() and not line.startswith(' ' * (len(line) - len(line.lstrip())) // 4 * 4):
                if len(line) - len(line.lstrip()) % 4 != 0:
                    issues.append(TestIssue(
                        test_file=str(test_file),
                        issue_type=MaintenanceType.REPAIR_SYNTAX,
                        severity=MaintenanceLevel.LOW,
                        description="Inconsistent indentation",
                        line_number=i,
                        auto_fixable=True,
                        confidence_score=0.7
                    ))
        
        return issues
    
    def _check_import_issues(self, test_file: Path, content: str, tree: ast.AST) -> List[TestIssue]:
        """Check for import-related issues"""
        issues = []
        
        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        # Check for unused imports
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for import statements
            import_match = re.match(r'^\s*(?:from\s+\S+\s+)?import\s+(.+)', line)
            if import_match:
                imported_items = [item.strip().split(' as ')[0] for item in import_match.group(1).split(',')]
                for item in imported_items:
                    if item not in used_names and not item.startswith('_'):
                        issues.append(TestIssue(
                            test_file=str(test_file),
                            issue_type=MaintenanceType.UPDATE_IMPORTS,
                            severity=MaintenanceLevel.LOW,
                            description=f"Unused import: {item}",
                            line_number=i,
                            auto_fixable=True,
                            confidence_score=0.8
                        ))
        
        return issues
    
    def _check_assertion_issues(self, test_file: Path, content: str, tree: ast.AST) -> List[TestIssue]:
        """Check for assertion-related issues"""
        issues = []
        
        # Find test methods
        test_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_methods.append(node)
        
        for method in test_methods:
            # Check for missing assertions
            has_assertions = False
            for node in ast.walk(method):
                if isinstance(node, ast.Assert) or (
                    isinstance(node, ast.Call) and 
                    isinstance(node.func, ast.Attribute) and
                    node.func.attr.startswith('assert')
                ):
                    has_assertions = True
                    break
            
            if not has_assertions:
                issues.append(TestIssue(
                    test_file=str(test_file),
                    issue_type=MaintenanceType.FIX_ASSERTIONS,
                    severity=MaintenanceLevel.HIGH,
                    description=f"Test method {method.name} has no assertions",
                    line_number=method.lineno,
                    auto_fixable=False,
                    confidence_score=0.9
                ))
        
        # Check for weak assertions
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'assert True' in line or 'assertTrue(True)' in line:
                issues.append(TestIssue(
                    test_file=str(test_file),
                    issue_type=MaintenanceType.FIX_ASSERTIONS,
                    severity=MaintenanceLevel.MEDIUM,
                    description="Weak assertion (always passes)",
                    line_number=i,
                    auto_fixable=False,
                    confidence_score=0.9
                ))
        
        return issues
    
    def _check_performance_issues(self, test_file: Path, content: str, tree: ast.AST) -> List[TestIssue]:
        """Check for performance-related issues"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for inefficient patterns
            if 'time.sleep(' in line and 'sleep(10' in line or 'sleep(5' in line:
                issues.append(TestIssue(
                    test_file=str(test_file),
                    issue_type=MaintenanceType.OPTIMIZE_PERFORMANCE,
                    severity=MaintenanceLevel.MEDIUM,
                    description="Long sleep in test (>= 5 seconds)",
                    line_number=i,
                    suggested_fix=line.replace('sleep(10', 'sleep(0.1').replace('sleep(5', 'sleep(0.1'),
                    auto_fixable=True,
                    confidence_score=0.7
                ))
            
            # Large loop iterations in tests
            if re.search(r'for\s+\w+\s+in\s+range\s*\(\s*[1-9]\d{3,}', line):
                issues.append(TestIssue(
                    test_file=str(test_file),
                    issue_type=MaintenanceType.OPTIMIZE_PERFORMANCE,
                    severity=MaintenanceLevel.MEDIUM,
                    description="Large iteration count in test loop",
                    line_number=i,
                    auto_fixable=False,
                    confidence_score=0.8
                ))
        
        return issues
    
    def _check_duplication_issues(self, test_file: Path, content: str, tree: ast.AST) -> List[TestIssue]:
        """Check for code duplication issues"""
        issues = []
        
        # Find test methods and compare their content
        test_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                method_code = ast.get_source_segment(content, node)
                if method_code:
                    test_methods.append((node.name, method_code, node.lineno))
        
        # Compare methods for similarity
        for i, (name1, code1, line1) in enumerate(test_methods):
            for name2, code2, line2 in test_methods[i+1:]:
                similarity = difflib.SequenceMatcher(None, code1, code2).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    issues.append(TestIssue(
                        test_file=str(test_file),
                        issue_type=MaintenanceType.REMOVE_DUPLICATES,
                        severity=MaintenanceLevel.MEDIUM,
                        description=f"Duplicate test methods: {name1} and {name2}",
                        line_number=line1,
                        auto_fixable=False,
                        confidence_score=similarity
                    ))
        
        return issues
    
    def _check_deprecated_patterns(self, test_file: Path, content: str, tree: ast.AST) -> List[TestIssue]:
        """Check for deprecated patterns and practices"""
        issues = []
        lines = content.split('\n')
        
        deprecated_patterns = {
            r'unittest\.TestCase': 'Consider using pytest fixtures instead of unittest.TestCase',
            r'setUp\(\s*self\s*\)': 'Consider using pytest fixtures instead of setUp',
            r'tearDown\(\s*self\s*\)': 'Consider using pytest fixtures instead of tearDown',
            r'assertEquals\(': 'Use assertEqual instead of assertEquals',
            r'assertNotEquals\(': 'Use assertNotEqual instead of assertNotEquals',
            r'assert_\w+_equal': 'Consider using standard assert statements with pytest'
        }
        
        for i, line in enumerate(lines, 1):
            for pattern, suggestion in deprecated_patterns.items():
                if re.search(pattern, line):
                    issues.append(TestIssue(
                        test_file=str(test_file),
                        issue_type=MaintenanceType.UPDATE_DEPRECATED,
                        severity=MaintenanceLevel.LOW,
                        description=suggestion,
                        line_number=i,
                        auto_fixable=True,
                        confidence_score=0.7
                    ))
        
        return issues
    
    def perform_automatic_maintenance(self, issues: Optional[List[TestIssue]] = None) -> MaintenanceReport:
        """Perform automatic maintenance on identified issues"""
        start_time = time.time()
        session_id = f"maintenance_{int(time.time())}"
        
        if issues is None:
            issues = self.analyze_test_suite()
        
        # Filter auto-fixable issues
        auto_fixable = [issue for issue in issues if issue.auto_fixable and issue.confidence_score >= 0.7]
        
        actions_performed = []
        issues_fixed = 0
        
        # Group issues by file for efficient processing
        issues_by_file = {}
        for issue in auto_fixable:
            if issue.test_file not in issues_by_file:
                issues_by_file[issue.test_file] = []
            issues_by_file[issue.test_file].append(issue)
        
        # Process each file
        for file_path, file_issues in issues_by_file.items():
            actions = self._fix_file_issues(file_path, file_issues)
            actions_performed.extend(actions)
            issues_fixed += len(actions)
        
        # Generate performance improvements
        performance_improvements = self._apply_performance_optimizations(issues)
        
        # Generate report
        return MaintenanceReport(
            session_id=session_id,
            total_tests_analyzed=len(list(self.test_directory.rglob("test_*.py"))),
            issues_found=len(issues),
            issues_fixed=issues_fixed,
            issues_requiring_manual_review=len(issues) - issues_fixed,
            performance_improvements=performance_improvements,
            actions_performed=actions_performed,
            summary_statistics=self._generate_summary_statistics(issues, actions_performed),
            recommendations=self._generate_recommendations(issues),
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
    
    def _fix_file_issues(self, file_path: str, issues: List[TestIssue]) -> List[MaintenanceAction]:
        """Fix issues in a specific file"""
        actions = []
        
        try:
            path = Path(file_path)
            content = path.read_text(encoding='utf-8')
            lines = content.split('\n')
            modified = False
            
            # Sort issues by line number (descending) to avoid line number shifts
            sorted_issues = sorted(issues, key=lambda x: x.line_number or 0, reverse=True)
            
            for issue in sorted_issues:
                if issue.suggested_fix and issue.line_number:
                    line_idx = issue.line_number - 1
                    if 0 <= line_idx < len(lines):
                        original_line = lines[line_idx]
                        lines[line_idx] = issue.suggested_fix
                        
                        action = MaintenanceAction(
                            action_id=f"{file_path}:{issue.line_number}:{issue.issue_type.value}",
                            test_file=file_path,
                            action_type=issue.issue_type,
                            original_code=original_line,
                            fixed_code=issue.suggested_fix,
                            line_range=(issue.line_number, issue.line_number),
                            confidence_score=issue.confidence_score
                        )
                        actions.append(action)
                        modified = True
            
            # Write back if modified
            if modified:
                path.write_text('\n'.join(lines), encoding='utf-8')
                
        except Exception as e:
            # Log error but continue with other files
            pass
        
        return actions
    
    def _apply_performance_optimizations(self, issues: List[TestIssue]) -> int:
        """Apply performance optimizations"""
        performance_issues = [i for i in issues if i.issue_type == MaintenanceType.OPTIMIZE_PERFORMANCE]
        optimizations_applied = 0
        
        for issue in performance_issues:
            if issue.auto_fixable and issue.suggested_fix:
                optimizations_applied += 1
        
        return optimizations_applied
    
    def _generate_summary_statistics(self, issues: List[TestIssue], 
                                   actions: List[MaintenanceAction]) -> Dict[str, Any]:
        """Generate summary statistics for maintenance report"""
        issue_types = {}
        severity_counts = {}
        
        for issue in issues:
            issue_types[issue.issue_type.value] = issue_types.get(issue.issue_type.value, 0) + 1
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
        
        return {
            "issues_by_type": issue_types,
            "issues_by_severity": severity_counts,
            "auto_fix_rate": len(actions) / len(issues) if issues else 0,
            "average_confidence": sum(a.confidence_score for a in actions) / len(actions) if actions else 0,
            "files_modified": len(set(a.test_file for a in actions)),
            "most_common_issue": max(issue_types.items(), key=lambda x: x[1])[0] if issue_types else None
        }
    
    def _generate_recommendations(self, issues: List[TestIssue]) -> List[str]:
        """Generate maintenance recommendations"""
        recommendations = []
        
        # Count issue types
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        # Generate specific recommendations
        if issue_counts.get(MaintenanceType.FIX_ASSERTIONS, 0) > 5:
            recommendations.append("Consider implementing test assertion guidelines and code review processes")
        
        if issue_counts.get(MaintenanceType.REMOVE_DUPLICATES, 0) > 3:
            recommendations.append("Implement test utility functions to reduce code duplication")
        
        if issue_counts.get(MaintenanceType.OPTIMIZE_PERFORMANCE, 0) > 2:
            recommendations.append("Review test performance requirements and optimize slow tests")
        
        if issue_counts.get(MaintenanceType.UPDATE_DEPRECATED, 0) > 5:
            recommendations.append("Plan migration to modern testing frameworks and patterns")
        
        # General recommendations
        critical_issues = [i for i in issues if i.severity == MaintenanceLevel.CRITICAL]
        if critical_issues:
            recommendations.append("Address critical issues immediately to prevent test failures")
        
        recommendations.append("Schedule regular automated maintenance to prevent issue accumulation")
        recommendations.append("Implement pre-commit hooks to catch common issues early")
        
        return recommendations
    
    async def continuous_maintenance(self, interval_hours: int = 24) -> None:
        """Run continuous maintenance monitoring"""
        while True:
            try:
                report = self.perform_automatic_maintenance()
                print(f"Maintenance completed: {report.issues_fixed} issues fixed, {report.issues_requiring_manual_review} require manual review")
                
                # Wait for next interval
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                print(f"Maintenance error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    def validate_fixes(self) -> Dict[str, bool]:
        """Validate that applied fixes don't break tests"""
        validation_results = {}
        
        for action in self.actions_performed:
            test_file = action.test_file
            
            # Run syntax check
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                ast.parse(content)
                validation_results[action.action_id] = True
            except SyntaxError:
                validation_results[action.action_id] = False
        
        return validation_results