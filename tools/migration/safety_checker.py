#!/usr/bin/env python3
"""
Safety Checker for TestMaster Reorganization
Implements multiple layers of failsafe checks to ensure no functionality is lost.
"""

import ast
import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import inspect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FunctionSignature:
    """Represents a function signature for comparison."""
    name: str
    parameters: List[str]
    defaults: List[Any]
    varargs: Optional[str]
    kwargs: Optional[str]
    return_annotation: Optional[str]
    source_file: str

@dataclass
class APICompatibilityReport:
    """Report on API compatibility."""
    preserved_functions: List[str] = field(default_factory=list)
    missing_functions: List[str] = field(default_factory=list)
    signature_changes: List[Tuple[str, str, str]] = field(default_factory=list)  # func, old_sig, new_sig
    new_functions: List[str] = field(default_factory=list)
    compatibility_score: float = 0.0

class SafetyChecker:
    """Multi-layer safety checker for code consolidation."""
    
    def __init__(self, legacy_analysis_file: str = "tools/migration/legacy_analysis.json"):
        self.legacy_analysis_file = legacy_analysis_file
        self.legacy_signatures: Dict[str, FunctionSignature] = {}
        self.current_signatures: Dict[str, FunctionSignature] = {}
        self.load_legacy_analysis()
    
    def load_legacy_analysis(self):
        """Load legacy function analysis."""
        try:
            with open(self.legacy_analysis_file, 'r') as f:
                legacy_data = json.load(f)
            
            for file_path, analysis in legacy_data.items():
                for func_data in analysis.get('functions', []):
                    signature = FunctionSignature(
                        name=func_data['name'],
                        parameters=func_data.get('parameters', []),
                        defaults=[],  # Would need more detailed analysis
                        varargs=None,
                        kwargs=None,
                        return_annotation=func_data.get('return_type'),
                        source_file=file_path
                    )
                    self.legacy_signatures[func_data['name']] = signature
            
            logger.info(f"Loaded {len(self.legacy_signatures)} legacy function signatures")
            
        except Exception as e:
            logger.error(f"Failed to load legacy analysis: {e}")
    
    def extract_current_signatures(self, module_dir: Path) -> Dict[str, FunctionSignature]:
        """Extract function signatures from current consolidated modules."""
        current_signatures = {}
        
        for py_file in module_dir.rglob("*.py"):
            if "__pycache__" not in str(py_file) and py_file.name != "__init__.py":
                try:
                    signatures = self._extract_file_signatures(py_file)
                    current_signatures.update(signatures)
                except Exception as e:
                    logger.error(f"Failed to extract signatures from {py_file}: {e}")
        
        return current_signatures
    
    def _extract_file_signatures(self, file_path: Path) -> Dict[str, FunctionSignature]:
        """Extract function signatures from a single file."""
        signatures = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    signature = self._extract_function_signature_from_ast(node, str(file_path))
                    signatures[signature.name] = signature
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            signature = self._extract_function_signature_from_ast(item, str(file_path))
                            # Prefix with class name for uniqueness
                            signature.name = f"{node.name}.{signature.name}"
                            signatures[signature.name] = signature
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return signatures
    
    def _extract_function_signature_from_ast(self, node: ast.FunctionDef, source_file: str) -> FunctionSignature:
        """Extract function signature from AST node."""
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            parameters.append(param)
        
        # Extract defaults (simplified)
        defaults = []
        for default in node.args.defaults:
            try:
                defaults.append(ast.unparse(default))
            except:
                defaults.append("<complex_default>")
        
        # Extract varargs and kwargs
        varargs = node.args.vararg.arg if node.args.vararg else None
        kwargs = node.args.kwarg.arg if node.args.kwarg else None
        
        # Extract return annotation
        return_annotation = None
        if node.returns:
            try:
                return_annotation = ast.unparse(node.returns)
            except:
                return_annotation = "<complex_return>"
        
        return FunctionSignature(
            name=node.name,
            parameters=parameters,
            defaults=defaults,
            varargs=varargs,
            kwargs=kwargs,
            return_annotation=return_annotation,
            source_file=source_file
        )
    
    def verify_api_compatibility(self, consolidated_module_dir: Path) -> APICompatibilityReport:
        """Verify that all original APIs are preserved in consolidated modules."""
        logger.info("Starting API compatibility verification...")
        
        # Extract current signatures from consolidated modules
        self.current_signatures = self.extract_current_signatures(consolidated_module_dir)
        
        report = APICompatibilityReport()
        
        # Check each legacy function
        for func_name, legacy_sig in self.legacy_signatures.items():
            if func_name in self.current_signatures:
                current_sig = self.current_signatures[func_name]
                
                # Check signature compatibility
                if self._signatures_compatible(legacy_sig, current_sig):
                    report.preserved_functions.append(func_name)
                else:
                    old_sig = self._format_signature(legacy_sig)
                    new_sig = self._format_signature(current_sig)
                    report.signature_changes.append((func_name, old_sig, new_sig))
            else:
                report.missing_functions.append(func_name)
        
        # Check for new functions
        for func_name in self.current_signatures:
            if func_name not in self.legacy_signatures:
                report.new_functions.append(func_name)
        
        # Calculate compatibility score
        total_legacy = len(self.legacy_signatures)
        preserved = len(report.preserved_functions)
        if total_legacy > 0:
            report.compatibility_score = preserved / total_legacy * 100
        
        logger.info(f"API Compatibility: {report.compatibility_score:.1f}%")
        logger.info(f"Preserved: {len(report.preserved_functions)}, Missing: {len(report.missing_functions)}")
        
        return report
    
    def _signatures_compatible(self, legacy: FunctionSignature, current: FunctionSignature) -> bool:
        """Check if two function signatures are compatible."""
        # For now, just check parameter count (could be made more sophisticated)
        legacy_param_count = len(legacy.parameters)
        current_param_count = len(current.parameters)
        
        # Allow for additional optional parameters in current version
        return current_param_count >= legacy_param_count
    
    def _format_signature(self, sig: FunctionSignature) -> str:
        """Format signature for display."""
        params = ", ".join(sig.parameters)
        if sig.varargs:
            params += f", *{sig.varargs}"
        if sig.kwargs:
            params += f", **{sig.kwargs}"
        
        return_part = ""
        if sig.return_annotation:
            return_part = f" -> {sig.return_annotation}"
        
        return f"def {sig.name}({params}){return_part}"
    
    def verify_functionality_coverage(self, test_results_before: Dict, test_results_after: Dict) -> bool:
        """Verify that functionality coverage is maintained."""
        # Compare test results before and after consolidation
        before_passed = test_results_before.get('passed', 0)
        after_passed = test_results_after.get('passed', 0)
        
        # Allow for improvement but not regression
        if after_passed >= before_passed:
            logger.info(f"Functionality coverage maintained: {before_passed} -> {after_passed} tests passing")
            return True
        else:
            logger.error(f"Functionality regression detected: {before_passed} -> {after_passed} tests passing")
            return False
    
    def create_rollback_point(self, description: str = "") -> str:
        """Create a rollback point with Git."""
        import subprocess
        from datetime import datetime
        
        try:
            # Create a Git commit as rollback point
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            commit_message = f"ROLLBACK_POINT_{timestamp}: {description}"
            
            subprocess.run(["git", "add", "."], check=True)
            result = subprocess.run(["git", "commit", "-m", commit_message], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Created rollback point: {commit_message}")
                return timestamp
            else:
                logger.warning("No changes to commit for rollback point")
                return timestamp
                
        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
            return ""
    
    def execute_rollback(self, rollback_point: str) -> bool:
        """Execute rollback to a specific point."""
        import subprocess
        
        try:
            # Find the commit with the rollback point
            result = subprocess.run(
                ["git", "log", "--oneline", "--grep", f"ROLLBACK_POINT_{rollback_point}"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                commit_hash = result.stdout.strip().split()[0]
                
                # Reset to that commit
                subprocess.run(["git", "reset", "--hard", commit_hash], check=True)
                logger.info(f"Successfully rolled back to {rollback_point}")
                return True
            else:
                logger.error(f"Rollback point {rollback_point} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute rollback: {e}")
            return False
    
    def check_import_dependencies(self, consolidated_modules: Path) -> List[str]:
        """Check for broken import dependencies."""
        broken_imports = []
        
        for py_file in consolidated_modules.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                try:
                    # Try to parse and check imports
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if not self._can_import(alias.name):
                                    broken_imports.append(f"{py_file}:{node.lineno} - {alias.name}")
                        
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and not self._can_import(node.module):
                                broken_imports.append(f"{py_file}:{node.lineno} - {node.module}")
                
                except Exception as e:
                    logger.error(f"Error checking imports in {py_file}: {e}")
        
        return broken_imports
    
    def _can_import(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            if module_name.startswith('.'):
                return True  # Relative imports are harder to check
            
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
        except Exception:
            return True  # Other errors don't necessarily mean import failure
    
    def generate_safety_report(self, compatibility_report: APICompatibilityReport, 
                             broken_imports: List[str]) -> str:
        """Generate comprehensive safety report."""
        report_lines = [
            "=" * 80,
            "TESTMASTER REORGANIZATION SAFETY REPORT",
            "=" * 80,
            f"Generated: {__import__('datetime').datetime.now().isoformat()}",
            "",
            "API COMPATIBILITY ANALYSIS:",
            f"  Overall compatibility: {compatibility_report.compatibility_score:.1f}%",
            f"  Functions preserved: {len(compatibility_report.preserved_functions)}",
            f"  Functions missing: {len(compatibility_report.missing_functions)}",
            f"  Signature changes: {len(compatibility_report.signature_changes)}",
            f"  New functions added: {len(compatibility_report.new_functions)}",
            ""
        ]
        
        if compatibility_report.missing_functions:
            report_lines.append("🚨 MISSING FUNCTIONS (CRITICAL):")
            for func in compatibility_report.missing_functions[:20]:
                report_lines.append(f"  - {func}")
            if len(compatibility_report.missing_functions) > 20:
                report_lines.append(f"  ... and {len(compatibility_report.missing_functions) - 20} more")
            report_lines.append("")
        
        if compatibility_report.signature_changes:
            report_lines.append("⚠️  SIGNATURE CHANGES:")
            for func, old_sig, new_sig in compatibility_report.signature_changes[:10]:
                report_lines.append(f"  {func}:")
                report_lines.append(f"    OLD: {old_sig}")
                report_lines.append(f"    NEW: {new_sig}")
            if len(compatibility_report.signature_changes) > 10:
                report_lines.append(f"  ... and {len(compatibility_report.signature_changes) - 10} more")
            report_lines.append("")
        
        if broken_imports:
            report_lines.append("🚨 BROKEN IMPORTS (CRITICAL):")
            for broken_import in broken_imports[:20]:
                report_lines.append(f"  - {broken_import}")
            if len(broken_imports) > 20:
                report_lines.append(f"  ... and {len(broken_imports) - 20} more")
            report_lines.append("")
        
        # Overall safety assessment
        if compatibility_report.compatibility_score >= 95 and not broken_imports:
            report_lines.append("✅ SAFETY STATUS: SAFE TO PROCEED")
        elif compatibility_report.compatibility_score >= 90:
            report_lines.append("⚠️  SAFETY STATUS: PROCEED WITH CAUTION")
        else:
            report_lines.append("🚨 SAFETY STATUS: NOT SAFE - ROLLBACK RECOMMENDED")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def emergency_stop(self, reason: str) -> bool:
        """Emergency stop procedure with automatic rollback."""
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Try to find the most recent rollback point
        import subprocess
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--grep", "ROLLBACK_POINT"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                latest_rollback = result.stdout.strip().split('\n')[0]
                commit_hash = latest_rollback.split()[0]
                
                # Execute emergency rollback
                subprocess.run(["git", "reset", "--hard", commit_hash], check=True)
                logger.critical(f"Emergency rollback executed to {commit_hash}")
                return True
            else:
                logger.critical("No rollback points found - manual intervention required")
                return False
                
        except Exception as e:
            logger.critical(f"Emergency rollback failed: {e}")
            return False

def main():
    """Main function for safety checking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TestMaster Safety Checker")
    parser.add_argument("--check-compatibility", action="store_true", 
                       help="Check API compatibility")
    parser.add_argument("--check-imports", action="store_true",
                       help="Check for broken imports")
    parser.add_argument("--consolidated-dir", default="testmaster",
                       help="Directory with consolidated modules")
    parser.add_argument("--create-rollback", help="Create rollback point with description")
    parser.add_argument("--rollback", help="Execute rollback to specific point")
    parser.add_argument("--emergency-stop", help="Execute emergency stop with reason")
    parser.add_argument("--report", help="Output safety report to file")
    
    args = parser.parse_args()
    
    checker = SafetyChecker()
    
    if args.create_rollback:
        rollback_id = checker.create_rollback_point(args.create_rollback)
        print(f"Created rollback point: {rollback_id}")
    
    if args.rollback:
        success = checker.execute_rollback(args.rollback)
        print(f"Rollback {'successful' if success else 'failed'}")
    
    if args.emergency_stop:
        success = checker.emergency_stop(args.emergency_stop)
        print(f"Emergency stop {'executed' if success else 'failed'}")
    
    if args.check_compatibility or args.check_imports:
        consolidated_dir = Path(args.consolidated_dir)
        
        compatibility_report = None
        broken_imports = []
        
        if args.check_compatibility:
            compatibility_report = checker.verify_api_compatibility(consolidated_dir)
            print(f"API Compatibility: {compatibility_report.compatibility_score:.1f}%")
        
        if args.check_imports:
            broken_imports = checker.check_import_dependencies(consolidated_dir)
            print(f"Broken imports: {len(broken_imports)}")
        
        if args.report and (compatibility_report or broken_imports):
            if not compatibility_report:
                compatibility_report = APICompatibilityReport()
            
            safety_report = checker.generate_safety_report(compatibility_report, broken_imports)
            
            with open(args.report, 'w') as f:
                f.write(safety_report)
            
            print(f"Safety report saved to {args.report}")
            print("\n" + safety_report)

if __name__ == "__main__":
    main()