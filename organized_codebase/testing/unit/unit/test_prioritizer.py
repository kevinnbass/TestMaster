#!/usr/bin/env python3
"""
Automated Test Categorization and Prioritization System
Intelligently categorizes and prioritizes tests for optimal execution.

Features:
- Automatic test categorization (unit, integration, e2e, performance)
- Risk-based prioritization algorithm
- Historical failure analysis
- Critical path identification
- Smart test selection
"""

import ast
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import logging
import hashlib
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Test categories."""
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "e2e"
    PERFORMANCE = "performance"
    SMOKE = "smoke"
    REGRESSION = "regression"
    SECURITY = "security"
    EDGE_CASE = "edge_case"
    BOUNDARY = "boundary"
    STRESS = "stress"


class Priority(Enum):
    """Test priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    TRIVIAL = 5


@dataclass
class TestMetadata:
    """Metadata for a test."""
    name: str
    file_path: Path
    category: TestCategory
    priority: Priority
    execution_time: float
    failure_rate: float
    last_failure: Optional[datetime]
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    complexity: int = 1
    coverage_impact: float = 0.0
    risk_score: float = 0.0
    flakiness_score: float = 0.0
    business_criticality: int = 5  # 1-10 scale
    
    def calculate_priority_score(self) -> float:
        """Calculate numerical priority score."""
        score = 0.0
        
        # Priority weight (lower is better)
        score += self.priority.value * 10
        
        # Risk score weight
        score += self.risk_score * 5
        
        # Failure rate weight
        score += self.failure_rate * 8
        
        # Business criticality (inverted - higher is better)
        score -= self.business_criticality * 3
        
        # Coverage impact
        score -= self.coverage_impact * 2
        
        # Flakiness penalty
        score += self.flakiness_score * 4
        
        # Recent failure boost
        if self.last_failure:
            days_since_failure = (datetime.now() - self.last_failure).days
            if days_since_failure < 7:
                score -= 10  # Recent failures get priority
        
        return score


class TestCategorizer:
    """Categorizes tests based on their characteristics."""
    
    def __init__(self):
        self.category_patterns = {
            TestCategory.UNIT: [
                r"test_.*_unit",
                r"test_single_.*",
                r"test_.*_function",
                r"test_.*_method"
            ],
            TestCategory.INTEGRATION: [
                r"test_.*_integration",
                r"test_.*_api",
                r"test_.*_database",
                r"test_.*_service"
            ],
            TestCategory.END_TO_END: [
                r"test_.*_e2e",
                r"test_.*_end_to_end",
                r"test_.*_workflow",
                r"test_.*_scenario"
            ],
            TestCategory.PERFORMANCE: [
                r"test_.*_performance",
                r"test_.*_speed",
                r"test_.*_benchmark",
                r"test_.*_load"
            ],
            TestCategory.SECURITY: [
                r"test_.*_security",
                r"test_.*_auth",
                r"test_.*_permission",
                r"test_.*_vulnerability"
            ],
            TestCategory.EDGE_CASE: [
                r"test_.*_edge",
                r"test_.*_corner",
                r"test_.*_special"
            ],
            TestCategory.BOUNDARY: [
                r"test_.*_boundary",
                r"test_.*_limit",
                r"test_.*_range"
            ],
            TestCategory.STRESS: [
                r"test_.*_stress",
                r"test_.*_capacity",
                r"test_.*_scale"
            ]
        }
    
    def categorize_test(self, test_name: str, test_ast: Optional[ast.FunctionDef] = None) -> TestCategory:
        """Categorize a test based on its name and content."""
        test_name_lower = test_name.lower()
        
        # Check patterns
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.match(pattern, test_name_lower):
                    return category
        
        # Analyze test content if AST provided
        if test_ast:
            category = self._analyze_test_content(test_ast)
            if category:
                return category
        
        # Check for common keywords
        if "mock" in test_name_lower or "stub" in test_name_lower:
            return TestCategory.UNIT
        elif "db" in test_name_lower or "database" in test_name_lower:
            return TestCategory.INTEGRATION
        elif "ui" in test_name_lower or "browser" in test_name_lower:
            return TestCategory.END_TO_END
        elif "load" in test_name_lower or "concurrent" in test_name_lower:
            return TestCategory.PERFORMANCE
        
        # Default to unit test
        return TestCategory.UNIT
    
    def _analyze_test_content(self, test_ast: ast.FunctionDef) -> Optional[TestCategory]:
        """Analyze test content to determine category."""
        # Count different types of operations
        db_ops = 0
        api_calls = 0
        ui_ops = 0
        mocks = 0
        assertions = 0
        
        for node in ast.walk(test_ast):
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'attr'):
                    attr = node.func.attr.lower()
                    
                    # Database operations
                    if any(db_word in attr for db_word in ['query', 'insert', 'update', 'delete', 'commit']):
                        db_ops += 1
                    
                    # API calls
                    elif any(api_word in attr for api_word in ['get', 'post', 'put', 'request', 'fetch']):
                        api_calls += 1
                    
                    # UI operations
                    elif any(ui_word in attr for ui_word in ['click', 'type', 'navigate', 'find_element']):
                        ui_ops += 1
                    
                    # Mocking
                    elif 'mock' in attr or 'patch' in attr:
                        mocks += 1
                    
                    # Assertions
                    elif 'assert' in attr:
                        assertions += 1
        
        # Determine category based on operation counts
        if ui_ops > 0:
            return TestCategory.END_TO_END
        elif db_ops > 0 or api_calls > 2:
            return TestCategory.INTEGRATION
        elif mocks > 2:
            return TestCategory.UNIT
        
        return None


class TestPrioritizer:
    """Prioritizes tests based on various factors."""
    
    def __init__(self, history_file: Optional[Path] = None):
        self.categorizer = TestCategorizer()
        self.test_metadata: Dict[str, TestMetadata] = {}
        self.execution_history: List[Dict] = []
        self.failure_patterns: Dict[str, List[str]] = defaultdict(list)
        self.history_file = history_file or Path(".testmaster_history.json")
        
        self.load_history()
    
    def analyze_test_file(self, file_path: Path) -> List[TestMetadata]:
        """Analyze a test file and extract metadata."""
        tests = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    metadata = self._extract_test_metadata(node, file_path)
                    tests.append(metadata)
                    self.test_metadata[metadata.name] = metadata
                
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                            metadata = self._extract_test_metadata(item, file_path, node.name)
                            tests.append(metadata)
                            self.test_metadata[f"{node.name}.{metadata.name}"] = metadata
        
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
        
        return tests
    
    def _extract_test_metadata(self, node: ast.FunctionDef, file_path: Path, 
                              class_name: Optional[str] = None) -> TestMetadata:
        """Extract metadata from test AST node."""
        full_name = f"{class_name}.{node.name}" if class_name else node.name
        
        # Categorize test
        category = self.categorizer.categorize_test(node.name, node)
        
        # Determine priority based on category and name
        priority = self._determine_priority(node.name, category)
        
        # Extract tags from docstring
        tags = self._extract_tags(node)
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        
        # Get historical data
        history = self._get_test_history(full_name)
        
        metadata = TestMetadata(
            name=full_name,
            file_path=file_path,
            category=category,
            priority=priority,
            execution_time=history.get('avg_execution_time', 0.1),
            failure_rate=history.get('failure_rate', 0.0),
            last_failure=history.get('last_failure'),
            tags=tags,
            complexity=complexity,
            coverage_impact=history.get('coverage_impact', 0.0),
            risk_score=self._calculate_risk_score(category, complexity, history),
            flakiness_score=history.get('flakiness_score', 0.0),
            business_criticality=self._determine_business_criticality(node.name, tags)
        )
        
        return metadata
    
    def _determine_priority(self, test_name: str, category: TestCategory) -> Priority:
        """Determine test priority."""
        test_name_lower = test_name.lower()
        
        # Critical keywords
        if any(word in test_name_lower for word in ['critical', 'core', 'essential', 'auth', 'security']):
            return Priority.CRITICAL
        
        # Category-based priority
        if category in [TestCategory.SECURITY, TestCategory.END_TO_END]:
            return Priority.HIGH
        elif category in [TestCategory.INTEGRATION, TestCategory.SMOKE]:
            return Priority.MEDIUM
        elif category in [TestCategory.PERFORMANCE, TestCategory.STRESS]:
            return Priority.MEDIUM
        elif category == TestCategory.UNIT:
            return Priority.LOW
        
        # Edge cases and boundaries are lower priority
        if category in [TestCategory.EDGE_CASE, TestCategory.BOUNDARY]:
            return Priority.LOW
        
        return Priority.MEDIUM
    
    def _extract_tags(self, node: ast.FunctionDef) -> Set[str]:
        """Extract tags from test docstring or decorators."""
        tags = set()
        
        # Check docstring for tags
        docstring = ast.get_docstring(node)
        if docstring:
            # Look for @tag or #tag patterns
            tag_matches = re.findall(r'[@#](\w+)', docstring)
            tags.update(tag_matches)
        
        # Check decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                tags.add(decorator.id)
            elif isinstance(decorator, ast.Call) and hasattr(decorator.func, 'id'):
                tags.add(decorator.func.id)
        
        return tags
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _calculate_risk_score(self, category: TestCategory, complexity: int, history: Dict) -> float:
        """Calculate risk score for test."""
        risk = 0.0
        
        # Category risk
        category_risks = {
            TestCategory.SECURITY: 10,
            TestCategory.END_TO_END: 8,
            TestCategory.INTEGRATION: 6,
            TestCategory.PERFORMANCE: 5,
            TestCategory.UNIT: 3,
            TestCategory.EDGE_CASE: 4,
            TestCategory.BOUNDARY: 4,
            TestCategory.STRESS: 7
        }
        risk += category_risks.get(category, 5)
        
        # Complexity risk
        risk += min(complexity / 2, 5)
        
        # Historical failure risk
        risk += history.get('failure_rate', 0) * 10
        
        # Recent changes risk
        if history.get('recently_changed', False):
            risk += 3
        
        return min(risk, 20)  # Cap at 20
    
    def _determine_business_criticality(self, test_name: str, tags: Set[str]) -> int:
        """Determine business criticality of test."""
        criticality = 5  # Default medium
        
        test_name_lower = test_name.lower()
        
        # Critical business functions
        critical_keywords = ['payment', 'auth', 'security', 'order', 'checkout', 'user', 'account']
        if any(keyword in test_name_lower for keyword in critical_keywords):
            criticality = 9
        
        # Check tags
        if 'critical' in tags or 'business_critical' in tags:
            criticality = 10
        elif 'important' in tags:
            criticality = 8
        elif 'low_priority' in tags or 'trivial' in tags:
            criticality = 2
        
        return criticality
    
    def _get_test_history(self, test_name: str) -> Dict:
        """Get historical data for test."""
        # This would normally query a database or file
        # For now, return mock data
        return {
            'avg_execution_time': 0.5,
            'failure_rate': 0.05,
            'last_failure': None,
            'coverage_impact': 2.5,
            'flakiness_score': 0.1,
            'recently_changed': False
        }
    
    def prioritize_tests(self, tests: List[TestMetadata], 
                        time_budget: Optional[float] = None,
                        risk_tolerance: str = "medium") -> List[TestMetadata]:
        """Prioritize tests based on various factors."""
        # Calculate priority scores
        for test in tests:
            test.risk_score = test.calculate_priority_score()
        
        # Sort by priority score (lower is better)
        prioritized = sorted(tests, key=lambda t: t.calculate_priority_score())
        
        # Apply time budget if specified
        if time_budget:
            selected = []
            total_time = 0
            
            for test in prioritized:
                if total_time + test.execution_time <= time_budget:
                    selected.append(test)
                    total_time += test.execution_time
                elif test.priority == Priority.CRITICAL:
                    # Always include critical tests
                    selected.append(test)
                    total_time += test.execution_time
            
            return selected
        
        # Apply risk tolerance
        if risk_tolerance == "low":
            # Run more tests when risk tolerance is low
            return prioritized
        elif risk_tolerance == "high":
            # Run only critical and high priority tests
            return [t for t in prioritized if t.priority.value <= 2]
        else:
            # Medium risk - balanced approach
            return prioritized[:int(len(prioritized) * 0.7)]
    
    def get_test_suite(self, category: Optional[TestCategory] = None,
                      priority: Optional[Priority] = None,
                      tags: Optional[Set[str]] = None) -> List[TestMetadata]:
        """Get filtered test suite."""
        tests = list(self.test_metadata.values())
        
        if category:
            tests = [t for t in tests if t.category == category]
        
        if priority:
            tests = [t for t in tests if t.priority == priority]
        
        if tags:
            tests = [t for t in tests if tags.intersection(t.tags)]
        
        return tests
    
    def generate_execution_plan(self, tests: List[TestMetadata]) -> Dict[str, Any]:
        """Generate optimal execution plan."""
        plan = {
            "phases": [],
            "total_time": 0,
            "test_count": len(tests),
            "parallel_groups": []
        }
        
        # Group tests by category and priority
        groups = defaultdict(list)
        for test in tests:
            groups[(test.category, test.priority)].append(test)
        
        # Create execution phases
        phase_order = [
            Priority.CRITICAL,
            Priority.HIGH,
            Priority.MEDIUM,
            Priority.LOW,
            Priority.TRIVIAL
        ]
        
        for priority in phase_order:
            phase_tests = []
            for (category, p), tests_in_group in groups.items():
                if p == priority:
                    phase_tests.extend(tests_in_group)
            
            if phase_tests:
                phase_time = sum(t.execution_time for t in phase_tests)
                plan["phases"].append({
                    "priority": priority.name,
                    "tests": [t.name for t in phase_tests],
                    "count": len(phase_tests),
                    "estimated_time": phase_time
                })
                plan["total_time"] += phase_time
        
        # Identify tests that can run in parallel
        plan["parallel_groups"] = self._identify_parallel_groups(tests)
        
        return plan
    
    def _identify_parallel_groups(self, tests: List[TestMetadata]) -> List[List[str]]:
        """Identify tests that can run in parallel."""
        # Group tests that don't share dependencies
        parallel_groups = []
        
        # Simple grouping by category for now
        category_groups = defaultdict(list)
        for test in tests:
            if test.category == TestCategory.UNIT:
                category_groups["unit"].append(test.name)
            elif test.category in [TestCategory.EDGE_CASE, TestCategory.BOUNDARY]:
                category_groups["edge"].append(test.name)
        
        for group in category_groups.values():
            if len(group) > 1:
                # Split into chunks of 5 for parallel execution
                for i in range(0, len(group), 5):
                    parallel_groups.append(group[i:i+5])
        
        return parallel_groups
    
    def update_test_results(self, test_name: str, passed: bool, execution_time: float):
        """Update test results for future prioritization."""
        if test_name not in self.test_metadata:
            return
        
        metadata = self.test_metadata[test_name]
        
        # Update execution time (moving average)
        metadata.execution_time = (metadata.execution_time * 0.7 + execution_time * 0.3)
        
        # Update failure rate
        if not passed:
            metadata.failure_rate = min(metadata.failure_rate + 0.1, 1.0)
            metadata.last_failure = datetime.now()
            self.failure_patterns[test_name].append(datetime.now().isoformat())
        else:
            metadata.failure_rate = max(metadata.failure_rate - 0.02, 0.0)
        
        # Save to history
        self.save_history()
    
    def save_history(self):
        """Save execution history to file."""
        history_data = {
            "updated_at": datetime.now().isoformat(),
            "tests": {}
        }
        
        for test_name, metadata in self.test_metadata.items():
            history_data["tests"][test_name] = {
                "execution_time": metadata.execution_time,
                "failure_rate": metadata.failure_rate,
                "last_failure": metadata.last_failure.isoformat() if metadata.last_failure else None,
                "flakiness_score": metadata.flakiness_score,
                "coverage_impact": metadata.coverage_impact
            }
        
        with open(self.history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def load_history(self):
        """Load execution history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    history_data = json.load(f)
                
                # Update test metadata with historical data
                for test_name, data in history_data.get("tests", {}).items():
                    if test_name in self.test_metadata:
                        metadata = self.test_metadata[test_name]
                        metadata.execution_time = data.get("execution_time", 0.1)
                        metadata.failure_rate = data.get("failure_rate", 0.0)
                        if data.get("last_failure"):
                            metadata.last_failure = datetime.fromisoformat(data["last_failure"])
                        metadata.flakiness_score = data.get("flakiness_score", 0.0)
                        metadata.coverage_impact = data.get("coverage_impact", 0.0)
                
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
    
    def generate_report(self, tests: List[TestMetadata]) -> str:
        """Generate prioritization report."""
        report_lines = [
            "=" * 60,
            "TEST PRIORITIZATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            f"Total tests: {len(tests)}",
            ""
        ]
        
        # Category breakdown
        category_counts = Counter(t.category for t in tests)
        report_lines.append("CATEGORIES:")
        for category, count in category_counts.most_common():
            report_lines.append(f"  {category.value}: {count} tests")
        
        # Priority breakdown
        priority_counts = Counter(t.priority for t in tests)
        report_lines.append("\nPRIORITIES:")
        for priority in Priority:
            count = priority_counts.get(priority, 0)
            report_lines.append(f"  {priority.name}: {count} tests")
        
        # Top priority tests
        top_tests = sorted(tests, key=lambda t: t.calculate_priority_score())[:10]
        report_lines.append("\nTOP PRIORITY TESTS:")
        for i, test in enumerate(top_tests, 1):
            report_lines.append(
                f"  {i}. {test.name} ({test.category.value}, {test.priority.name}, "
                f"risk: {test.risk_score:.1f})"
            )
        
        # High risk tests
        high_risk = [t for t in tests if t.risk_score > 10]
        if high_risk:
            report_lines.append(f"\nHIGH RISK TESTS ({len(high_risk)}):")
            for test in high_risk[:5]:
                report_lines.append(f"  - {test.name} (risk: {test.risk_score:.1f})")
        
        # Flaky tests
        flaky = [t for t in tests if t.flakiness_score > 0.3]
        if flaky:
            report_lines.append(f"\nFLAKY TESTS ({len(flaky)}):")
            for test in flaky[:5]:
                report_lines.append(f"  - {test.name} (flakiness: {test.flakiness_score:.2f})")
        
        # Execution time estimate
        total_time = sum(t.execution_time for t in tests)
        report_lines.append(f"\nEXECUTION TIME ESTIMATE:")
        report_lines.append(f"  Total: {total_time:.1f} seconds")
        report_lines.append(f"  Average: {total_time/len(tests):.2f} seconds per test")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)


def main():
    """CLI for test prioritization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Categorization and Prioritization")
    parser.add_argument("--test-dir", required=True, help="Directory containing test files")
    parser.add_argument("--analyze", action="store_true", help="Analyze and categorize tests")
    parser.add_argument("--prioritize", action="store_true", help="Generate prioritized test list")
    parser.add_argument("--time-budget", type=float, help="Time budget in seconds")
    parser.add_argument("--risk-tolerance", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--priority", help="Filter by priority")
    parser.add_argument("--execution-plan", action="store_true", help="Generate execution plan")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize prioritizer
    prioritizer = TestPrioritizer()
    
    # Analyze test files
    test_dir = Path(args.test_dir)
    all_tests = []
    
    print(f"Analyzing tests in {test_dir}...")
    for test_file in test_dir.rglob("test_*.py"):
        tests = prioritizer.analyze_test_file(test_file)
        all_tests.extend(tests)
    
    print(f"Found {len(all_tests)} tests")
    
    if args.analyze:
        # Show categorization results
        category_counts = Counter(t.category for t in all_tests)
        print("\nTest Categories:")
        for category, count in category_counts.most_common():
            print(f"  {category.value}: {count} tests")
        
        priority_counts = Counter(t.priority for t in all_tests)
        print("\nTest Priorities:")
        for priority in Priority:
            count = priority_counts.get(priority, 0)
            print(f"  {priority.name}: {count} tests")
    
    if args.prioritize:
        # Prioritize tests
        prioritized = prioritizer.prioritize_tests(
            all_tests,
            time_budget=args.time_budget,
            risk_tolerance=args.risk_tolerance
        )
        
        print(f"\nPrioritized {len(prioritized)} tests")
        print("\nTop 10 priority tests:")
        for i, test in enumerate(prioritized[:10], 1):
            print(f"  {i}. {test.name} ({test.category.value}, {test.priority.name})")
        
        if args.time_budget:
            total_time = sum(t.execution_time for t in prioritized)
            print(f"\nEstimated execution time: {total_time:.1f} seconds")
            print(f"Time budget: {args.time_budget} seconds")
    
    if args.execution_plan:
        # Generate execution plan
        tests_to_plan = all_tests
        
        if args.category:
            category = TestCategory(args.category)
            tests_to_plan = prioritizer.get_test_suite(category=category)
        
        if args.priority:
            priority = Priority[args.priority.upper()]
            tests_to_plan = prioritizer.get_test_suite(priority=priority)
        
        plan = prioritizer.generate_execution_plan(tests_to_plan)
        
        print("\nExecution Plan:")
        print("=" * 40)
        for i, phase in enumerate(plan["phases"], 1):
            print(f"\nPhase {i}: {phase['priority']}")
            print(f"  Tests: {phase['count']}")
            print(f"  Time: {phase['estimated_time']:.1f}s")
        
        print(f"\nTotal execution time: {plan['total_time']:.1f} seconds")
        
        if plan["parallel_groups"]:
            print(f"\nParallel execution groups: {len(plan['parallel_groups'])}")
    
    if args.report:
        # Generate detailed report
        report = prioritizer.generate_report(all_tests)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)


if __name__ == "__main__":
    main()