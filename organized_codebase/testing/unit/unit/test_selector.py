"""
Intelligent Test Selector for TestMaster
ML-powered test prioritization and selection based on code changes
"""

import ast
import hashlib
import json
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import re


class SelectionStrategy(Enum):
    """Test selection strategies"""
    IMPACT = "impact"  # Based on change impact
    RISK = "risk"  # Based on failure risk
    COVERAGE = "coverage"  # Based on code coverage
    HISTORY = "history"  # Based on failure history
    SMART = "smart"  # ML-based combination


@dataclass
class TestCase:
    """Represents a test case"""
    id: str
    name: str
    file_path: str
    coverage: Set[str]  # Files/functions covered
    execution_time: float
    failure_rate: float
    last_failure: Optional[str]
    priority: int = 0
    dependencies: List[str] = None


@dataclass
class CodeChange:
    """Represents a code change"""
    file_path: str
    changed_lines: List[int]
    changed_functions: List[str]
    change_type: str  # add, modify, delete
    risk_score: float = 0.0


@dataclass
class SelectionResult:
    """Result of test selection"""
    selected_tests: List[TestCase]
    excluded_tests: List[TestCase]
    estimated_time: float
    coverage_score: float
    risk_coverage: float
    selection_reason: Dict[str, str]


class TestImpactAnalyzer:
    """Analyzes impact of code changes on tests"""
    
    def __init__(self):
        self.dependency_graph = {}
        self.test_coverage_map = {}
        self.function_call_graph = {}
        
    def analyze_change_impact(self, change: CodeChange) -> Dict[str, float]:
        """Analyze impact of a code change"""
        impact = {
            'direct': 0.0,
            'indirect': 0.0,
            'risk': 0.0
        }
        
        # Direct impact: tests that cover changed code
        impact['direct'] = len(self._get_direct_tests(change))
        
        # Indirect impact: tests affected through dependencies
        impact['indirect'] = len(self._get_indirect_tests(change))
        
        # Risk assessment based on change type and location
        impact['risk'] = self._calculate_risk(change)
        
        return impact
    
    def _get_direct_tests(self, change: CodeChange) -> Set[str]:
        """Get tests directly covering changed code"""
        direct_tests = set()
        
        for test_id, coverage in self.test_coverage_map.items():
            if change.file_path in coverage:
                # Check if specific functions are covered
                for func in change.changed_functions:
                    if func in coverage.get(change.file_path, []):
                        direct_tests.add(test_id)
                        
        return direct_tests
    
    def _get_indirect_tests(self, change: CodeChange) -> Set[str]:
        """Get tests indirectly affected through dependencies"""
        indirect_tests = set()
        
        # Find all functions that depend on changed functions
        affected_funcs = set()
        for func in change.changed_functions:
            affected_funcs.update(self._get_dependent_functions(func))
            
        # Find tests covering affected functions
        for test_id, coverage in self.test_coverage_map.items():
            for func in affected_funcs:
                if func in coverage.get(change.file_path, []):
                    indirect_tests.add(test_id)
                    
        return indirect_tests
    
    def _get_dependent_functions(self, func: str) -> Set[str]:
        """Get functions that depend on given function"""
        dependents = set()
        
        for caller, callees in self.function_call_graph.items():
            if func in callees:
                dependents.add(caller)
                # Recursive for transitive dependencies
                dependents.update(self._get_dependent_functions(caller))
                
        return dependents
    
    def _calculate_risk(self, change: CodeChange) -> float:
        """Calculate risk score for a change"""
        risk = 0.0
        
        # Risk factors
        if change.change_type == 'delete':
            risk += 0.5
        elif change.change_type == 'modify':
            risk += 0.3
        else:  # add
            risk += 0.2
            
        # Critical file patterns
        critical_patterns = [
            r'auth', r'security', r'payment', r'database',
            r'config', r'api', r'critical', r'core'
        ]
        
        for pattern in critical_patterns:
            if re.search(pattern, change.file_path, re.IGNORECASE):
                risk += 0.2
                break
                
        # Number of changed lines factor
        if len(change.changed_lines) > 100:
            risk += 0.3
        elif len(change.changed_lines) > 50:
            risk += 0.2
        elif len(change.changed_lines) > 20:
            risk += 0.1
            
        return min(risk, 1.0)  # Cap at 1.0


class TestSelector:
    """Main test selection engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.impact_analyzer = TestImpactAnalyzer()
        self.test_database = {}
        self.execution_history = []
        self.ml_model = None  # Would load ML model
        
    def select_tests(self, changes: List[CodeChange], all_tests: List[TestCase],
                    strategy: SelectionStrategy = SelectionStrategy.SMART,
                    time_budget: Optional[float] = None) -> SelectionResult:
        """Select tests based on code changes"""
        
        if strategy == SelectionStrategy.IMPACT:
            return self._select_by_impact(changes, all_tests, time_budget)
        elif strategy == SelectionStrategy.RISK:
            return self._select_by_risk(changes, all_tests, time_budget)
        elif strategy == SelectionStrategy.COVERAGE:
            return self._select_by_coverage(changes, all_tests, time_budget)
        elif strategy == SelectionStrategy.HISTORY:
            return self._select_by_history(changes, all_tests, time_budget)
        elif strategy == SelectionStrategy.SMART:
            return self._select_smart(changes, all_tests, time_budget)
        else:
            return self._select_all(all_tests)
    
    def _select_by_impact(self, changes: List[CodeChange], 
                         tests: List[TestCase], budget: Optional[float]) -> SelectionResult:
        """Select tests based on change impact"""
        test_scores = {}
        
        for test in tests:
            score = 0.0
            for change in changes:
                impact = self.impact_analyzer.analyze_change_impact(change)
                
                # Check if test covers changed code
                if change.file_path in test.coverage:
                    score += impact['direct'] * 1.0
                    score += impact['indirect'] * 0.5
                    score += impact['risk'] * 0.3
                    
            test_scores[test.id] = score
        
        # Sort by score
        sorted_tests = sorted(tests, key=lambda t: test_scores[t.id], reverse=True)
        
        return self._apply_budget(sorted_tests, budget, "impact-based")
    
    def _select_by_risk(self, changes: List[CodeChange],
                       tests: List[TestCase], budget: Optional[float]) -> SelectionResult:
        """Select tests based on risk assessment"""
        # Calculate overall risk
        total_risk = sum(change.risk_score for change in changes)
        
        # Prioritize tests by failure rate and coverage of risky areas
        test_scores = {}
        for test in tests:
            score = test.failure_rate * 0.5
            
            # Bonus for covering risky changes
            for change in changes:
                if change.file_path in test.coverage:
                    score += change.risk_score * 0.5
                    
            test_scores[test.id] = score
        
        sorted_tests = sorted(tests, key=lambda t: test_scores[t.id], reverse=True)
        
        return self._apply_budget(sorted_tests, budget, "risk-based")
    
    def _select_by_coverage(self, changes: List[CodeChange],
                           tests: List[TestCase], budget: Optional[float]) -> SelectionResult:
        """Select tests to maximize coverage"""
        selected = []
        covered = set()
        remaining = tests.copy()
        
        while remaining and (budget is None or 
                           sum(t.execution_time for t in selected) < budget):
            # Find test that covers most uncovered code
            best_test = None
            best_new_coverage = 0
            
            for test in remaining:
                new_coverage = len(test.coverage - covered)
                if new_coverage > best_new_coverage:
                    best_test = test
                    best_new_coverage = new_coverage
                    
            if best_test and best_new_coverage > 0:
                selected.append(best_test)
                covered.update(best_test.coverage)
                remaining.remove(best_test)
            else:
                break
                
        excluded = [t for t in tests if t not in selected]
        
        return SelectionResult(
            selected_tests=selected,
            excluded_tests=excluded,
            estimated_time=sum(t.execution_time for t in selected),
            coverage_score=len(covered) / len(set.union(*[t.coverage for t in tests])) if tests else 0,
            risk_coverage=0.0,
            selection_reason={"strategy": "coverage-maximization"}
        )
    
    def _select_by_history(self, changes: List[CodeChange],
                          tests: List[TestCase], budget: Optional[float]) -> SelectionResult:
        """Select tests based on historical failure patterns"""
        # Prioritize recently failed tests
        test_scores = {}
        
        for test in tests:
            score = 0.0
            
            # Recent failure weight
            if test.last_failure:
                days_since_failure = self._days_since(test.last_failure)
                score += max(0, 10 - days_since_failure) / 10.0
                
            # Failure rate weight
            score += test.failure_rate
            
            # Execution time penalty (prefer faster tests)
            score -= test.execution_time / 100.0
            
            test_scores[test.id] = score
        
        sorted_tests = sorted(tests, key=lambda t: test_scores[t.id], reverse=True)
        
        return self._apply_budget(sorted_tests, budget, "history-based")
    
    def _select_smart(self, changes: List[CodeChange],
                     tests: List[TestCase], budget: Optional[float]) -> SelectionResult:
        """Smart selection using ML and multiple strategies"""
        
        # Combine multiple strategies
        impact_result = self._select_by_impact(changes, tests, None)
        risk_result = self._select_by_risk(changes, tests, None)
        history_result = self._select_by_history(changes, tests, None)
        
        # Score each test based on multiple strategies
        test_scores = defaultdict(float)
        
        for i, test in enumerate(impact_result.selected_tests):
            test_scores[test.id] += (len(tests) - i) * 0.3
            
        for i, test in enumerate(risk_result.selected_tests):
            test_scores[test.id] += (len(tests) - i) * 0.3
            
        for i, test in enumerate(history_result.selected_tests):
            test_scores[test.id] += (len(tests) - i) * 0.2
            
        # Coverage bonus
        covered = set()
        for test in tests:
            new_coverage = len(test.coverage - covered)
            if new_coverage > 0:
                test_scores[test.id] += new_coverage * 0.2
                covered.update(test.coverage)
        
        # Sort by combined score
        sorted_tests = sorted(tests, key=lambda t: test_scores[t.id], reverse=True)
        
        return self._apply_budget(sorted_tests, budget, "ml-smart")
    
    def _apply_budget(self, sorted_tests: List[TestCase], 
                     budget: Optional[float], strategy: str) -> SelectionResult:
        """Apply time budget to test selection"""
        if budget is None:
            selected = sorted_tests
            excluded = []
        else:
            selected = []
            total_time = 0.0
            
            for test in sorted_tests:
                if total_time + test.execution_time <= budget:
                    selected.append(test)
                    total_time += test.execution_time
                    
            excluded = [t for t in sorted_tests if t not in selected]
        
        # Calculate metrics
        all_coverage = set.union(*[t.coverage for t in sorted_tests]) if sorted_tests else set()
        selected_coverage = set.union(*[t.coverage for t in selected]) if selected else set()
        
        return SelectionResult(
            selected_tests=selected,
            excluded_tests=excluded,
            estimated_time=sum(t.execution_time for t in selected),
            coverage_score=len(selected_coverage) / len(all_coverage) if all_coverage else 0,
            risk_coverage=self._calculate_risk_coverage(selected),
            selection_reason={"strategy": strategy, "budget": budget}
        )
    
    def _calculate_risk_coverage(self, tests: List[TestCase]) -> float:
        """Calculate risk coverage score"""
        if not tests:
            return 0.0
            
        # Weight by failure rate
        weighted_sum = sum(t.failure_rate for t in tests)
        max_possible = len(tests)
        
        return min(weighted_sum / max_possible, 1.0) if max_possible > 0 else 0.0
    
    def _days_since(self, date_str: str) -> int:
        """Calculate days since date (simplified)"""
        # Would parse date and calculate difference
        return 7  # Placeholder
    
    def _select_all(self, tests: List[TestCase]) -> SelectionResult:
        """Select all tests (fallback)"""
        return SelectionResult(
            selected_tests=tests,
            excluded_tests=[],
            estimated_time=sum(t.execution_time for t in tests),
            coverage_score=1.0,
            risk_coverage=1.0,
            selection_reason={"strategy": "all"}
        )