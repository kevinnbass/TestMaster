#!/usr/bin/env python3
"""
Test Quality Analyzer for TestMaster

Consolidates functionality from:
- independent_test_verifier.py

Provides comprehensive test quality analysis without modifying tests.
"""

import os
import sys
import json
import ast
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import QualityAnalyzer, VerificationResult, VerificationConfig

class TestQualityAnalyzer(QualityAnalyzer):
    """
    Independent test quality analyzer that evaluates tests without modification.
    
    Provides detailed quality reports and improvement suggestions.
    """
    
    def __init__(self, 
                 mode: str = "auto",
                 model: str = None,
                 api_key: Optional[str] = None,
                 config: Optional[VerificationConfig] = None):
        """
        Initialize quality analyzer.
        
        Args:
            mode: AI mode ("provider", "sdk", "template", "auto")
            model: AI model to use
            api_key: API key for AI services
            config: Verification configuration
        """
        super().__init__(config)
        self.mode = mode
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.provider = None
        self.client = None
        
        # Initialize AI provider if available
        if mode != "template":
            try:
                if mode == "auto":
                    self._auto_initialize()
                elif mode == "provider":
                    self._init_provider_mode()
                elif mode == "sdk":
                    self._init_sdk_mode()
            except Exception as e:
                print(f"AI initialization failed: {e}, using template mode")
                self.mode = "template"
    
    def _auto_initialize(self):
        """Auto-detect best available AI provider."""
        try:
            self._init_provider_mode()
            self.mode = "provider"
            return
        except:
            pass
        
        try:
            self._init_sdk_mode()
            self.mode = "sdk"
            return
        except:
            pass
        
        self.mode = "template"
    
    def _init_provider_mode(self):
        """Initialize using GeminiProvider."""
        from multi_coder_analysis.llm_providers.gemini_provider import GeminiProvider
        
        if self.model is None:
            self.model = "models/gemini-2.5-pro"
        
        self.provider = GeminiProvider(model=self.model, api_key=self.api_key)
    
    def _init_sdk_mode(self):
        """Initialize using direct Google GenAI SDK."""
        import google.generativeai as genai
        
        if self.model is None:
            self.model = "models/gemini-2.5-pro"
        
        genai.configure(api_key=self.api_key)
        self.client = genai
    
    def verify_test(self, test_file: Path, module_file: Optional[Path] = None) -> VerificationResult:
        """Verify test quality without modification."""
        start_time = time.time()
        
        try:
            print(f"\n{'='*60}")
            print(f"Quality Analysis: {test_file.name}")
            print('='*60)
            
            # Comprehensive quality analysis
            analysis = self.analyze_test_quality(test_file, module_file)
            
            if "error" in analysis:
                return VerificationResult(
                    success=False,
                    test_file=str(test_file),
                    error_message=analysis["error"],
                    execution_time=time.time() - start_time
                )
            
            # Extract results
            structure = analysis["structure"]
            quality_score = analysis["quality_score"]
            issues = analysis["issues"]
            suggestions = analysis["suggestions"]
            
            # Additional analysis
            coverage_score = self._analyze_coverage(test_file, module_file)
            complexity_analysis = self._analyze_complexity(test_file)
            
            print(f"Quality Analysis Results:")
            print(f"  Overall quality score: {quality_score:.1f}/100")
            print(f"  Test methods: {structure.get('test_methods', 0)}")
            print(f"  Assertions: {structure.get('assertions', 0)}")
            print(f"  Coverage score: {coverage_score:.1f}/100")
            print(f"  Issues found: {len(issues)}")
            print(f"  Suggestions: {len(suggestions)}")
            
            if issues:
                print(f"\nIssues identified:")
                for issue in issues[:5]:
                    print(f"  - {issue}")
                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more")
            
            if suggestions:
                print(f"\nImprovement suggestions:")
                for suggestion in suggestions[:3]:
                    print(f"  - {suggestion}")
                if len(suggestions) > 3:
                    print(f"  ... and {len(suggestions) - 3} more")
            
            execution_time = time.time() - start_time
            
            result = VerificationResult(
                success=True,
                test_file=str(test_file),
                module_file=str(module_file) if module_file else None,
                quality_score=quality_score,
                test_count=structure.get('test_methods', 0),
                assertion_count=structure.get('assertions', 0),
                coverage_score=coverage_score,
                issues=issues,
                suggestions=suggestions,
                execution_time=execution_time
            )
            
            self.update_stats(result)
            return result
            
        except Exception as e:
            error_result = VerificationResult(
                success=False,
                test_file=str(test_file),
                module_file=str(module_file) if module_file else None,
                error_message=f"Analysis error: {e}",
                execution_time=time.time() - start_time
            )
            self.update_stats(error_result)
            return error_result
    
    def _analyze_coverage(self, test_file: Path, module_file: Optional[Path] = None) -> float:
        """Analyze test coverage by comparing test and module code."""
        if not module_file or not module_file.exists():
            return 50.0  # Default score when module not available
        
        try:
            test_code = test_file.read_text(encoding='utf-8')
            module_code = module_file.read_text(encoding='utf-8')
            
            # Extract functions and classes from module
            module_tree = ast.parse(module_code)
            module_functions = set()
            module_classes = set()
            
            for node in ast.walk(module_tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    module_functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    module_classes.add(node.name)
                    # Add public methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                            module_functions.add(f"{node.name}.{item.name}")
            
            # Check coverage in test code
            covered_functions = set()
            covered_classes = set()
            
            # Simple coverage analysis based on name mentions
            test_lower = test_code.lower()
            for func in module_functions:
                if func.lower() in test_lower:
                    covered_functions.add(func)
            
            for cls in module_classes:
                if cls.lower() in test_lower:
                    covered_classes.add(cls)
            
            # Calculate coverage score
            total_items = len(module_functions) + len(module_classes)
            covered_items = len(covered_functions) + len(covered_classes)
            
            if total_items == 0:
                return 100.0
            
            coverage = (covered_items / total_items) * 100
            return min(coverage, 100.0)
            
        except Exception as e:
            print(f"Coverage analysis error: {e}")
            return 50.0
    
    def _analyze_complexity(self, test_file: Path) -> Dict[str, Any]:
        """Analyze test complexity metrics."""
        try:
            test_code = test_file.read_text(encoding='utf-8')
            tree = ast.parse(test_code)
            
            complexity = {
                "cyclomatic_complexity": 0,
                "nesting_depth": 0,
                "test_method_length": [],
                "total_lines": len(test_code.splitlines())
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    # Calculate cyclomatic complexity for test method
                    method_complexity = 1
                    max_depth = 0
                    current_depth = 0
                    
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                            method_complexity += 1
                        
                        # Rough nesting depth calculation
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                            current_depth += 1
                            max_depth = max(max_depth, current_depth)
                    
                    complexity["cyclomatic_complexity"] += method_complexity
                    complexity["nesting_depth"] = max(complexity["nesting_depth"], max_depth)
                    
                    # Method length (line count)
                    if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                        method_length = node.end_lineno - node.lineno
                        complexity["test_method_length"].append(method_length)
            
            return complexity
            
        except Exception as e:
            print(f"Complexity analysis error: {e}")
            return {}
    
    def generate_quality_report(self, test_files: List[Path], output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive quality report for multiple test files."""
        print(f"\n{'='*60}")
        print(f"GENERATING QUALITY REPORT")
        print('='*60)
        print(f"Analyzing {len(test_files)} test files...")
        
        results = []
        for test_file in test_files:
            result = self.verify_test(test_file)
            results.append(result)
        
        # Aggregate statistics
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            avg_quality = sum(r.quality_score for r in successful_results) / len(successful_results)
            avg_coverage = sum(r.coverage_score for r in successful_results) / len(successful_results)
            total_tests = sum(r.test_count for r in successful_results)
            total_assertions = sum(r.assertion_count for r in successful_results)
            
            # Quality distribution
            quality_ranges = {
                "excellent (90-100)": len([r for r in successful_results if r.quality_score >= 90]),
                "good (70-89)": len([r for r in successful_results if 70 <= r.quality_score < 90]),
                "fair (50-69)": len([r for r in successful_results if 50 <= r.quality_score < 70]),
                "poor (0-49)": len([r for r in successful_results if r.quality_score < 50])
            }
        else:
            avg_quality = 0
            avg_coverage = 0
            total_tests = 0
            total_assertions = 0
            quality_ranges = {}
        
        # Compile all issues and suggestions
        all_issues = []
        all_suggestions = []
        
        for result in successful_results:
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)
        
        # Count issue frequencies
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        # Create report
        report = {
            "summary": {
                "total_files": len(test_files),
                "successful_analysis": len(successful_results),
                "failed_analysis": len(results) - len(successful_results),
                "average_quality_score": avg_quality,
                "average_coverage_score": avg_coverage,
                "total_test_methods": total_tests,
                "total_assertions": total_assertions
            },
            "quality_distribution": quality_ranges,
            "common_issues": dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "common_suggestions": dict(sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "detailed_results": [
                {
                    "file": r.test_file,
                    "quality_score": r.quality_score,
                    "coverage_score": r.coverage_score,
                    "test_count": r.test_count,
                    "assertion_count": r.assertion_count,
                    "issues": r.issues,
                    "suggestions": r.suggestions
                } for r in successful_results
            ]
        }
        
        # Save report if requested
        if output_file:
            output_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f"Quality report saved to: {output_file}")
        
        # Print summary
        self._print_quality_report_summary(report)
        
        return report
    
    def _print_quality_report_summary(self, report: Dict[str, Any]):
        """Print quality report summary."""
        summary = report["summary"]
        
        print(f"\n{'='*60}")
        print("QUALITY REPORT SUMMARY")
        print('='*60)
        print(f"Files analyzed: {summary['total_files']}")
        print(f"Successful: {summary['successful_analysis']}")
        print(f"Failed: {summary['failed_analysis']}")
        print(f"Average quality score: {summary['average_quality_score']:.1f}")
        print(f"Average coverage score: {summary['average_coverage_score']:.1f}")
        print(f"Total test methods: {summary['total_test_methods']}")
        print(f"Total assertions: {summary['total_assertions']}")
        
        print(f"\nQuality Distribution:")
        for range_name, count in report["quality_distribution"].items():
            print(f"  {range_name}: {count} files")
        
        print(f"\nMost Common Issues:")
        for issue, count in list(report["common_issues"].items())[:5]:
            print(f"  {issue}: {count} files")
        
        print(f"\nTop Suggestions:")
        for suggestion, count in list(report["common_suggestions"].items())[:3]:
            print(f"  {suggestion}: {count} files")


def main():
    """Test the quality analyzer."""
    print("="*60)
    print("TestMaster Quality Analyzer")
    print("="*60)
    
    # Create analyzer
    config = VerificationConfig(
        quality_threshold=70.0,
        use_ai_analysis=False  # Independent analysis
    )
    
    analyzer = TestQualityAnalyzer(mode="template", config=config)
    
    # Find test files
    test_dir = Path("tests/unit")
    if not test_dir.exists():
        test_dir = Path("tests")
    
    if test_dir.exists():
        test_files = list(test_dir.glob("test_*.py"))[:10]  # Test first 10
        if test_files:
            print(f"Analyzing {len(test_files)} test files...")
            
            # Generate quality report
            report_file = Path("test_quality_report.json")
            report = analyzer.generate_quality_report(test_files, report_file)
            
        else:
            print("No test files found to analyze")
            return 1
    else:
        print("No test directory found")
        return 1
    
    # Print final stats
    analyzer.print_stats()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())