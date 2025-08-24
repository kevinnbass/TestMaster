#!/usr/bin/env python3
"""
Base Verification Classes for TestMaster

Provides foundation classes for all test verification systems.
"""

import os
import sys
import ast
import time
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
import threading

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Result of test verification."""
    success: bool
    test_file: str
    module_file: Optional[str] = None
    quality_score: float = 0.0
    test_count: int = 0
    assertion_count: int = 0
    coverage_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: Optional[str] = None

@dataclass
class VerificationConfig:
    """Configuration for test verification."""
    quality_threshold: float = 70.0
    min_test_count: int = 1
    min_assertion_count: int = 1
    require_docstrings: bool = True
    check_edge_cases: bool = True
    check_error_handling: bool = True
    healing_iterations: int = 5
    use_ai_analysis: bool = True
    rate_limit_rpm: int = 30

class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_call = time.time()

class BaseVerifier(ABC):
    """Base class for all test verifiers."""
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        """Initialize verifier with configuration."""
        self.config = config or VerificationConfig()
        self.rate_limiter = RateLimiter(self.config.rate_limit_rpm)
        self.stats = {
            "verifications_attempted": 0,
            "verifications_successful": 0,
            "verifications_failed": 0,
            "total_execution_time": 0.0,
            "total_quality_score": 0.0,
            "start_time": time.time()
        }
    
    @abstractmethod
    def verify_test(self, test_file: Path, module_file: Optional[Path] = None) -> VerificationResult:
        """Verify a single test file."""
        pass
    
    def analyze_test_structure(self, test_code: str) -> Dict[str, Any]:
        """Analyze the structure of test code."""
        try:
            tree = ast.parse(test_code)
            
            analysis = {
                "test_methods": 0,
                "test_classes": 0,
                "assertions": 0,
                "docstrings": 0,
                "imports": 0,
                "mocks": 0,
                "fixtures": 0,
                "error_handling": 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_"):
                        analysis["test_methods"] += 1
                        if ast.get_docstring(node):
                            analysis["docstrings"] += 1
                        
                        # Check for error handling
                        for child in ast.walk(node):
                            if isinstance(child, ast.ExceptHandler) or isinstance(child, ast.Raise):
                                analysis["error_handling"] += 1
                                break
                
                elif isinstance(node, ast.ClassDef):
                    if node.name.startswith("Test"):
                        analysis["test_classes"] += 1
                
                elif isinstance(node, ast.Assert):
                    analysis["assertions"] += 1
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis["imports"] += 1
                    # Check for mock imports
                    if hasattr(node, 'names'):
                        for alias in node.names:
                            if 'mock' in alias.name.lower():
                                analysis["mocks"] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing test structure: {e}")
            return {}
    
    def calculate_quality_score(self, analysis: Dict[str, Any], test_code: str = "") -> float:
        """Calculate quality score based on analysis."""
        try:
            score = 0.0
            
            # Base score for having tests
            if analysis.get("test_methods", 0) > 0:
                score += 20
            
            # Test method count (up to 30 points)
            test_count = analysis.get("test_methods", 0)
            score += min(test_count * 5, 30)
            
            # Assertion count (up to 25 points)
            assertion_count = analysis.get("assertions", 0)
            score += min(assertion_count * 2, 25)
            
            # Docstring coverage (up to 15 points)
            if test_count > 0:
                docstring_ratio = analysis.get("docstrings", 0) / test_count
                score += docstring_ratio * 15
            
            # Error handling (up to 10 points)
            if analysis.get("error_handling", 0) > 0:
                score += 10
            
            return min(score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def find_module_for_test(self, test_file: Path) -> Optional[Path]:
        """Find the source module corresponding to a test file."""
        test_name = test_file.stem
        if test_name.startswith("test_"):
            module_name = test_name[5:]  # Remove "test_" prefix
        else:
            module_name = test_name
        
        # Remove common suffixes
        module_name = module_name.replace("_intelligent", "")
        module_name = module_name.replace("_test", "")
        
        # Search for module
        base_dirs = ["multi_coder_analysis", "src", "testmaster", "."]
        
        for base_dir in base_dirs:
            base_path = Path(base_dir)
            if base_path.exists():
                # Direct match
                module_file = base_path / f"{module_name}.py"
                if module_file.exists():
                    return module_file
                
                # Recursive search
                for py_file in base_path.rglob(f"{module_name}.py"):
                    return py_file
        
        return None
    
    def update_stats(self, result: VerificationResult):
        """Update verification statistics."""
        self.stats["verifications_attempted"] += 1
        if result.success:
            self.stats["verifications_successful"] += 1
            self.stats["total_quality_score"] += result.quality_score
        else:
            self.stats["verifications_failed"] += 1
        self.stats["total_execution_time"] += result.execution_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        elapsed = time.time() - self.stats["start_time"]
        attempted = self.stats["verifications_attempted"]
        successful = self.stats["verifications_successful"]
        
        avg_quality = 0.0
        if successful > 0:
            avg_quality = self.stats["total_quality_score"] / successful
        
        return {
            **self.stats,
            "elapsed_time": elapsed,
            "success_rate": (successful / max(1, attempted)) * 100,
            "average_quality_score": avg_quality,
            "average_time_per_verification": (
                self.stats["total_execution_time"] / max(1, attempted)
            )
        }
    
    def print_stats(self):
        """Print verification statistics."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print("VERIFICATION STATISTICS")
        print('='*60)
        print(f"Attempted: {stats['verifications_attempted']}")
        print(f"Successful: {stats['verifications_successful']}")
        print(f"Failed: {stats['verifications_failed']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Average quality score: {stats['average_quality_score']:.1f}")
        print(f"Total time: {stats['elapsed_time']:.1f}s")
        print(f"Avg time per verification: {stats['average_time_per_verification']:.1f}s")

class SelfHealingVerifier(BaseVerifier):
    """Base class for self-healing verifiers."""
    
    def heal_syntax_errors(self, test_code: str, max_iterations: int = None) -> Optional[str]:
        """Attempt to heal syntax errors in test code."""
        if max_iterations is None:
            max_iterations = self.config.healing_iterations
        
        for iteration in range(max_iterations):
            try:
                # Check if syntax is valid
                ast.parse(test_code)
                return test_code  # No healing needed
            except SyntaxError as e:
                logger.info(f"Healing iteration {iteration + 1}: {e}")
                
                # Attempt to heal using AI if available
                if self.config.use_ai_analysis:
                    healed_code = self._heal_with_ai(test_code, str(e))
                    if healed_code:
                        test_code = healed_code
                        continue
                
                # Basic healing strategies
                healed_code = self._basic_syntax_healing(test_code, e)
                if healed_code:
                    test_code = healed_code
                else:
                    break
        
        return None  # Could not heal
    
    @abstractmethod
    def _heal_with_ai(self, test_code: str, error_message: str) -> Optional[str]:
        """Heal using AI (implemented by subclasses)."""
        pass
    
    def _basic_syntax_healing(self, test_code: str, error: SyntaxError) -> Optional[str]:
        """Basic syntax healing strategies."""
        lines = test_code.split('\n')
        
        try:
            error_line = error.lineno - 1
            if 0 <= error_line < len(lines):
                line = lines[error_line]
                
                # Common fixes
                if "invalid syntax" in str(error):
                    # Try adding missing colons
                    if any(keyword in line for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:']):
                        if not line.rstrip().endswith(':'):
                            lines[error_line] = line.rstrip() + ':'
                            return '\n'.join(lines)
                    
                    # Try fixing indentation
                    if line.strip() and not line[0].isspace():
                        lines[error_line] = '    ' + line
                        return '\n'.join(lines)
                
                # Try removing the problematic line as last resort
                if error_line > 0:
                    lines.pop(error_line)
                    return '\n'.join(lines)
        
        except Exception:
            pass
        
        return None

class QualityAnalyzer(BaseVerifier):
    """Base class for quality analysis verifiers."""
    
    def analyze_test_quality(self, test_file: Path, module_file: Optional[Path] = None) -> Dict[str, Any]:
        """Comprehensive quality analysis of test file."""
        try:
            test_code = test_file.read_text(encoding='utf-8')
            
            # Structural analysis
            structure = self.analyze_test_structure(test_code)
            
            # Quality metrics
            quality_score = self.calculate_quality_score(structure, test_code)
            
            # Issues and suggestions
            issues = self._identify_issues(structure, test_code)
            suggestions = self._generate_suggestions(structure, test_code, module_file)
            
            return {
                "structure": structure,
                "quality_score": quality_score,
                "issues": issues,
                "suggestions": suggestions,
                "test_file": str(test_file),
                "module_file": str(module_file) if module_file else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing test quality: {e}")
            return {
                "error": str(e),
                "test_file": str(test_file)
            }
    
    def _identify_issues(self, structure: Dict[str, Any], test_code: str) -> List[str]:
        """Identify issues in test code."""
        issues = []
        
        if structure.get("test_methods", 0) == 0:
            issues.append("No test methods found")
        
        if structure.get("assertions", 0) == 0:
            issues.append("No assertions found")
        
        if structure.get("docstrings", 0) == 0 and self.config.require_docstrings:
            issues.append("No docstrings found")
        
        if structure.get("error_handling", 0) == 0 and self.config.check_error_handling:
            issues.append("No error handling tests found")
        
        # Check for common anti-patterns
        if "pass" in test_code and "TODO" not in test_code:
            issues.append("Test methods with only 'pass' statement found")
        
        if structure.get("test_methods", 0) > 0 and structure.get("assertions", 0) == 0:
            issues.append("Test methods without assertions")
        
        return issues
    
    def _generate_suggestions(self, structure: Dict[str, Any], test_code: str, 
                           module_file: Optional[Path] = None) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if structure.get("test_methods", 0) < 3:
            suggestions.append("Consider adding more test methods for better coverage")
        
        if structure.get("assertions", 0) < structure.get("test_methods", 0) * 2:
            suggestions.append("Consider adding more assertions per test method")
        
        if structure.get("docstrings", 0) < structure.get("test_methods", 0):
            suggestions.append("Add docstrings to test methods for better documentation")
        
        if structure.get("error_handling", 0) == 0:
            suggestions.append("Add tests for error conditions and edge cases")
        
        if "mock" not in test_code.lower() and module_file:
            suggestions.append("Consider using mocks for external dependencies")
        
        return suggestions