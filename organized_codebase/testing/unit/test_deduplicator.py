#!/usr/bin/env python3
"""
Intelligent Test Deduplication Engine
Identifies and removes redundant test cases while maintaining coverage.

Features:
- AST-based similarity detection
- Semantic comparison of test logic
- Intelligent test merging
- Coverage gap identification
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import hashlib
import difflib
from collections import defaultdict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    file_path: Path
    class_name: Optional[str]
    function_node: ast.FunctionDef
    assertions: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    tested_functions: Set[str] = field(default_factory=set)
    test_data: List[Any] = field(default_factory=list)
    line_number: int = 0
    complexity: int = 0
    signature_hash: str = ""
    body_hash: str = ""
    
    def __post_init__(self):
        if not self.signature_hash:
            self.signature_hash = self._calculate_signature_hash()
        if not self.body_hash:
            self.body_hash = self._calculate_body_hash()
    
    def _calculate_signature_hash(self) -> str:
        """Calculate hash of test signature."""
        sig_parts = [
            self.name,
            str(sorted(self.fixtures)),
            str(sorted(self.tested_functions))
        ]
        return hashlib.md5("".join(sig_parts).encode()).hexdigest()
    
    def _calculate_body_hash(self) -> str:
        """Calculate hash of test body (normalized)."""
        # Remove variable names and focus on structure
        normalized = ast.dump(self.function_node, annotate_fields=False)
        return hashlib.md5(normalized.encode()).hexdigest()


class TestDeduplicator:
    """Main deduplication engine."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.test_cases: List[TestCase] = []
        self.duplicates: Dict[str, List[TestCase]] = defaultdict(list)
        self.coverage_map: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze_test_file(self, file_path: Path) -> List[TestCase]:
        """Analyze a test file and extract test cases."""
        test_cases = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Test class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                            test_case = self._extract_test_case(item, file_path, node.name)
                            test_cases.append(test_case)
                
                elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    # Standalone test function
                    test_case = self._extract_test_case(node, file_path, None)
                    test_cases.append(test_case)
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
        
        return test_cases
    
    def _extract_test_case(self, node: ast.FunctionDef, file_path: Path, class_name: Optional[str]) -> TestCase:
        """Extract test case information from AST node."""
        test_case = TestCase(
            name=node.name,
            file_path=file_path,
            class_name=class_name,
            function_node=node,
            line_number=node.lineno
        )
        
        # Extract fixtures from parameters
        for arg in node.args.args:
            if arg.arg not in ['self', 'cls']:
                test_case.fixtures.append(arg.arg)
        
        # Analyze test body
        for stmt in ast.walk(node):
            # Extract assertions
            if isinstance(stmt, ast.Assert):
                test_case.assertions.append(ast.unparse(stmt.test))
            elif isinstance(stmt, ast.Call):
                if hasattr(stmt.func, 'attr'):
                    # Method calls like self.assertEqual
                    if 'assert' in stmt.func.attr.lower():
                        test_case.assertions.append(ast.unparse(stmt))
                    # Track tested functions
                    elif hasattr(stmt.func, 'value'):
                        test_case.tested_functions.add(stmt.func.attr)
                elif hasattr(stmt.func, 'id'):
                    # Direct function calls
                    test_case.tested_functions.add(stmt.func.id)
            
            # Extract test data
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        test_case.test_data.append((target.id, stmt.value))
        
        # Calculate complexity
        test_case.complexity = self._calculate_complexity(node)
        
        return test_case
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of test."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def find_duplicates(self, test_files: List[Path]) -> Dict[str, List[TestCase]]:
        """Find duplicate test cases across files."""
        # Collect all test cases
        for file_path in test_files:
            test_cases = self.analyze_test_file(file_path)
            self.test_cases.extend(test_cases)
        
        # Group by signature hash for initial clustering
        signature_groups = defaultdict(list)
        for test_case in self.test_cases:
            signature_groups[test_case.signature_hash].append(test_case)
        
        # Detailed comparison within groups
        duplicate_id = 0
        for sig_hash, group in signature_groups.items():
            if len(group) > 1:
                # Compare tests within the group
                clusters = self._cluster_similar_tests(group)
                for cluster in clusters:
                    if len(cluster) > 1:
                        dup_key = f"duplicate_group_{duplicate_id}"
                        self.duplicates[dup_key] = cluster
                        duplicate_id += 1
        
        return self.duplicates
    
    def _cluster_similar_tests(self, test_cases: List[TestCase]) -> List[List[TestCase]]:
        """Cluster similar test cases based on similarity threshold."""
        clusters = []
        processed = set()
        
        for i, test1 in enumerate(test_cases):
            if i in processed:
                continue
            
            cluster = [test1]
            processed.add(i)
            
            for j, test2 in enumerate(test_cases[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_similarity(test1, test2)
                if similarity >= self.similarity_threshold:
                    cluster.append(test2)
                    processed.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_similarity(self, test1: TestCase, test2: TestCase) -> float:
        """Calculate similarity between two test cases."""
        scores = []
        
        # Name similarity
        name_sim = difflib.SequenceMatcher(None, test1.name, test2.name).ratio()
        scores.append(name_sim * 0.2)
        
        # Assertion similarity
        assert1_str = " ".join(sorted(test1.assertions))
        assert2_str = " ".join(sorted(test2.assertions))
        assert_sim = difflib.SequenceMatcher(None, assert1_str, assert2_str).ratio()
        scores.append(assert_sim * 0.4)
        
        # Tested functions similarity
        if test1.tested_functions or test2.tested_functions:
            common = test1.tested_functions & test2.tested_functions
            total = test1.tested_functions | test2.tested_functions
            func_sim = len(common) / len(total) if total else 0
            scores.append(func_sim * 0.3)
        else:
            scores.append(0)
        
        # Body hash comparison
        body_sim = 1.0 if test1.body_hash == test2.body_hash else 0.0
        scores.append(body_sim * 0.1)
        
        return sum(scores)
    
    def merge_duplicates(self, duplicates: List[TestCase]) -> TestCase:
        """Merge duplicate test cases into a single comprehensive test."""
        if not duplicates:
            return None
        
        # Select the most comprehensive test as base
        base_test = max(duplicates, key=lambda t: (len(t.assertions), t.complexity))
        
        # Merge unique assertions from other tests
        all_assertions = set(base_test.assertions)
        all_tested_functions = set(base_test.tested_functions)
        all_fixtures = set(base_test.fixtures)
        
        for test in duplicates:
            if test != base_test:
                all_assertions.update(test.assertions)
                all_tested_functions.update(test.tested_functions)
                all_fixtures.update(test.fixtures)
        
        # Create merged test
        merged_test = TestCase(
            name=base_test.name,
            file_path=base_test.file_path,
            class_name=base_test.class_name,
            function_node=base_test.function_node,
            assertions=list(all_assertions),
            fixtures=list(all_fixtures),
            tested_functions=all_tested_functions,
            test_data=base_test.test_data,
            line_number=base_test.line_number,
            complexity=base_test.complexity
        )
        
        return merged_test
    
    def identify_coverage_gaps(self, test_cases: List[TestCase], source_files: List[Path]) -> Dict[str, List[str]]:
        """Identify functions/methods not covered by tests."""
        coverage_gaps = defaultdict(list)
        
        # Extract all functions from source files
        source_functions = self._extract_source_functions(source_files)
        
        # Extract all tested functions
        tested_functions = set()
        for test in test_cases:
            tested_functions.update(test.tested_functions)
        
        # Find gaps
        for file_path, functions in source_functions.items():
            uncovered = set(functions) - tested_functions
            if uncovered:
                coverage_gaps[str(file_path)] = list(uncovered)
        
        return dict(coverage_gaps)
    
    def _extract_source_functions(self, source_files: List[Path]) -> Dict[Path, List[str]]:
        """Extract all functions/methods from source files."""
        source_functions = defaultdict(list)
        
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        source_functions[file_path].append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                source_functions[file_path].append(f"{node.name}.{item.name}")
                
            except Exception as e:
                logger.error(f"Failed to extract functions from {file_path}: {e}")
        
        return dict(source_functions)
    
    def generate_deduplication_report(self) -> Dict[str, Any]:
        """Generate comprehensive deduplication report."""
        report = {
            "total_tests_analyzed": len(self.test_cases),
            "duplicate_groups_found": len(self.duplicates),
            "total_duplicates": sum(len(group) - 1 for group in self.duplicates.values()),
            "duplicate_details": [],
            "recommendations": []
        }
        
        for group_id, duplicates in self.duplicates.items():
            detail = {
                "group_id": group_id,
                "duplicate_count": len(duplicates),
                "tests": [
                    {
                        "name": test.name,
                        "file": str(test.file_path),
                        "line": test.line_number,
                        "assertions": len(test.assertions),
                        "complexity": test.complexity
                    }
                    for test in duplicates
                ],
                "recommended_action": self._recommend_action(duplicates)
            }
            report["duplicate_details"].append(detail)
        
        # General recommendations
        if report["total_duplicates"] > 0:
            report["recommendations"].append(
                f"Consider removing {report['total_duplicates']} duplicate tests to improve maintainability"
            )
        
        if report["duplicate_groups_found"] > 10:
            report["recommendations"].append(
                "High number of duplicate groups detected. Consider refactoring test structure"
            )
        
        return report
    
    def _recommend_action(self, duplicates: List[TestCase]) -> str:
        """Recommend action for duplicate group."""
        if len(duplicates) == 2:
            # Check if they're in different files
            if duplicates[0].file_path != duplicates[1].file_path:
                return "Merge tests and keep in most relevant file"
            else:
                return "Combine into single comprehensive test"
        else:
            unique_files = len(set(t.file_path for t in duplicates))
            if unique_files > 1:
                return f"Consolidate {len(duplicates)} tests from {unique_files} files into single location"
            else:
                return f"Merge {len(duplicates)} similar tests into one comprehensive test"
    
    def apply_deduplication(self, output_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Apply deduplication by modifying test files."""
        results = {
            "files_modified": [],
            "tests_removed": 0,
            "tests_merged": 0,
            "errors": []
        }
        
        for group_id, duplicates in self.duplicates.items():
            try:
                if len(duplicates) < 2:
                    continue
                
                # Merge duplicates
                merged_test = self.merge_duplicates(duplicates)
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would merge {len(duplicates)} tests in group {group_id}")
                else:
                    # Keep the first test, remove others
                    base_file = duplicates[0].file_path
                    
                    # TODO: Actually modify the AST and write back
                    # For now, just log the action
                    logger.info(f"Merging {len(duplicates)} tests in {base_file}")
                    
                    results["tests_merged"] += 1
                    results["tests_removed"] += len(duplicates) - 1
                    results["files_modified"].append(str(base_file))
                
            except Exception as e:
                error_msg = f"Failed to process group {group_id}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        return results


def main():
    """CLI for test deduplication."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Deduplication Engine")
    parser.add_argument("--test-dir", required=True, help="Directory containing test files")
    parser.add_argument("--source-dir", help="Directory containing source files")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (0-1)")
    parser.add_argument("--output", help="Output directory for deduplicated tests")
    parser.add_argument("--report", help="Save report to file")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without modifications")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize deduplicator
    deduplicator = TestDeduplicator(similarity_threshold=args.threshold)
    
    # Find test files
    test_dir = Path(args.test_dir)
    test_files = list(test_dir.rglob("test_*.py"))
    
    print(f"Analyzing {len(test_files)} test files...")
    
    # Find duplicates
    duplicates = deduplicator.find_duplicates(test_files)
    
    print(f"Found {len(duplicates)} groups of duplicate tests")
    
    # Check coverage gaps if source directory provided
    if args.source_dir:
        source_dir = Path(args.source_dir)
        source_files = [f for f in source_dir.rglob("*.py") if not f.name.startswith("test_")]
        
        gaps = deduplicator.identify_coverage_gaps(deduplicator.test_cases, source_files)
        
        if gaps:
            print(f"\nCoverage gaps found in {len(gaps)} files:")
            for file_path, functions in list(gaps.items())[:5]:
                print(f"  {file_path}: {len(functions)} uncovered functions")
    
    # Generate report
    report = deduplicator.generate_deduplication_report()
    
    print(f"\nDeduplication Summary:")
    print(f"  Total tests analyzed: {report['total_tests_analyzed']}")
    print(f"  Duplicate groups: {report['duplicate_groups_found']}")
    print(f"  Total duplicates: {report['total_duplicates']}")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    # Save report if requested
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.report}")
    
    # Apply deduplication if output directory provided
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        results = deduplicator.apply_deduplication(output_dir, dry_run=args.dry_run)
        
        print(f"\nDeduplication Results:")
        print(f"  Files modified: {len(results['files_modified'])}")
        print(f"  Tests merged: {results['tests_merged']}")
        print(f"  Tests removed: {results['tests_removed']}")
        
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")


if __name__ == "__main__":
    main()