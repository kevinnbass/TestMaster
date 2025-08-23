#!/usr/bin/env python3
"""
Intelligence Module Validator - Agent A Hour 7
Comprehensive testing and validation system for activated intelligence modules

Validates all activated intelligence modules for functionality, compatibility,
and integration with the architecture framework.
"""

import logging
import importlib
import importlib.util
import sys
import traceback
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import ast
import inspect

# Import architecture framework
from core.architecture.architecture_integration import get_architecture_framework
from core.services.service_registry import get_service_registry


class ValidationStatus(Enum):
    """Validation test result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class TestType(Enum):
    """Type of validation test"""
    SYNTAX = "syntax"
    IMPORT = "import"
    INSTANTIATION = "instantiation"
    FUNCTIONALITY = "functionality"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"


@dataclass
class ValidationResult:
    """Result of module validation test"""
    module_name: str
    test_type: TestType
    status: ValidationStatus
    message: str
    execution_time: float = 0.0
    details: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.details is None:
            self.details = {}


@dataclass
class ModuleValidationReport:
    """Comprehensive validation report for a module"""
    module_name: str
    module_path: Path
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    error_tests: int
    overall_status: ValidationStatus
    results: List[ValidationResult]
    validation_time: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0


class IntelligenceModuleValidator:
    """
    Intelligence Module Validation System
    
    Provides comprehensive testing and validation of activated intelligence
    modules including syntax, imports, functionality, and integration tests.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Architecture components
        self.framework = get_architecture_framework()
        self.service_registry = get_service_registry()
        
        # Path configuration
        self.intelligence_base = Path("core/intelligence")
        self.test_results_dir = Path("tests/intelligence_validation")
        self.test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation configuration
        self.validation_config = {
            'enable_syntax_tests': True,
            'enable_import_tests': True,
            'enable_instantiation_tests': True,
            'enable_functionality_tests': True,
            'enable_integration_tests': True,
            'enable_performance_tests': True,
            'timeout_seconds': 30,
            'max_memory_mb': 256,
            'skip_heavy_tests': False
        }
        
        # Test tracking
        self.validation_reports: List[ModuleValidationReport] = []
        self.discovered_modules: List[Path] = []
        
        # Statistics
        self.stats = {
            'modules_tested': 0,
            'modules_passed': 0,
            'modules_failed': 0,
            'modules_warning': 0,
            'total_tests': 0,
            'total_passed': 0,
            'total_failed': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info("Intelligence Module Validator initialized")
    
    def discover_intelligence_modules(self) -> List[Path]:
        """Discover all activated intelligence modules"""
        modules = []
        
        if not self.intelligence_base.exists():
            self.logger.warning(f"Intelligence base path not found: {self.intelligence_base}")
            return modules
        
        try:
            # Find Python modules in intelligence directory
            for module_file in self.intelligence_base.rglob("*.py"):
                # Skip __init__.py and test files
                if (module_file.name.startswith("__") or 
                    module_file.name.startswith("test_") or
                    "test" in module_file.parts):
                    continue
                
                modules.append(module_file)
            
            self.discovered_modules = modules
            self.logger.info(f"Discovered {len(modules)} intelligence modules for validation")
            
        except Exception as e:
            self.logger.error(f"Failed to discover intelligence modules: {e}")
        
        return modules
    
    def validate_all_modules(self) -> List[ModuleValidationReport]:
        """Validate all discovered intelligence modules"""
        self.logger.info("Starting comprehensive intelligence module validation...")
        
        modules = self.discover_intelligence_modules()
        
        for module_path in modules:
            try:
                start_time = time.time()
                report = self._validate_single_module(module_path)
                validation_time = time.time() - start_time
                
                report.validation_time = validation_time
                self.validation_reports.append(report)
                
                # Update statistics
                self.stats['modules_tested'] += 1
                self.stats['total_tests'] += report.total_tests
                self.stats['total_passed'] += report.passed_tests
                self.stats['total_failed'] += report.failed_tests
                
                if report.overall_status == ValidationStatus.PASSED:
                    self.stats['modules_passed'] += 1
                elif report.overall_status == ValidationStatus.FAILED:
                    self.stats['modules_failed'] += 1
                elif report.overall_status == ValidationStatus.WARNING:
                    self.stats['modules_warning'] += 1
                
                self.logger.info(f"Validated {module_path.name}: {report.overall_status.value} "
                               f"({report.passed_tests}/{report.total_tests} tests passed)")
                
            except Exception as e:
                self.logger.error(f"Failed to validate module {module_path}: {e}")
                self.stats['modules_failed'] += 1
        
        # Generate comprehensive report
        self._generate_validation_summary()
        
        return self.validation_reports
    
    def _validate_single_module(self, module_path: Path) -> ModuleValidationReport:
        """Validate a single intelligence module"""
        module_name = module_path.stem
        results = []
        
        self.logger.debug(f"Validating module: {module_name}")
        
        # Test 1: Syntax Validation
        if self.validation_config['enable_syntax_tests']:
            results.append(self._test_syntax(module_path))
        
        # Test 2: Import Validation
        if self.validation_config['enable_import_tests']:
            results.append(self._test_imports(module_path))
        
        # Test 3: Module Loading
        if self.validation_config['enable_instantiation_tests']:
            results.extend(self._test_module_loading(module_path))
        
        # Test 4: Functionality Tests
        if self.validation_config['enable_functionality_tests']:
            results.extend(self._test_functionality(module_path))
        
        # Test 5: Integration Tests
        if self.validation_config['enable_integration_tests']:
            results.extend(self._test_integration(module_path))
        
        # Test 6: Performance Tests
        if (self.validation_config['enable_performance_tests'] and
            not self.validation_config['skip_heavy_tests']):
            results.extend(self._test_performance(module_path))
        
        # Calculate overall status
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == ValidationStatus.PASSED])
        failed_tests = len([r for r in results if r.status == ValidationStatus.FAILED])
        warning_tests = len([r for r in results if r.status == ValidationStatus.WARNING])
        error_tests = len([r for r in results if r.status == ValidationStatus.ERROR])
        
        # Determine overall status
        if error_tests > 0 or failed_tests > total_tests * 0.5:
            overall_status = ValidationStatus.FAILED
        elif warning_tests > 0 or failed_tests > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED
        
        return ModuleValidationReport(
            module_name=module_name,
            module_path=module_path,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            error_tests=error_tests,
            overall_status=overall_status,
            results=results,
            validation_time=0.0,  # Set by caller
            metadata={
                'file_size': module_path.stat().st_size,
                'last_modified': datetime.fromtimestamp(module_path.stat().st_mtime),
                'line_count': self._count_lines(module_path)
            }
        )
    
    def _test_syntax(self, module_path: Path) -> ValidationResult:
        """Test module syntax validity"""
        start_time = time.time()
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Compile to check syntax
            compile(content, str(module_path), 'exec')
            
            execution_time = time.time() - start_time
            return ValidationResult(
                module_name=module_path.stem,
                test_type=TestType.SYNTAX,
                status=ValidationStatus.PASSED,
                message="Syntax validation passed",
                execution_time=execution_time
            )
        
        except SyntaxError as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                module_name=module_path.stem,
                test_type=TestType.SYNTAX,
                status=ValidationStatus.FAILED,
                message=f"Syntax error: {e}",
                execution_time=execution_time,
                details={'line': e.lineno, 'offset': e.offset}
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                module_name=module_path.stem,
                test_type=TestType.SYNTAX,
                status=ValidationStatus.ERROR,
                message=f"Syntax test error: {e}",
                execution_time=execution_time
            )
    
    def _test_imports(self, module_path: Path) -> ValidationResult:
        """Test module import dependencies"""
        start_time = time.time()
        
        try:
            # Parse module to find imports
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            missing_imports = []
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Test each import
            for imp in imports:
                try:
                    # Skip relative imports and standard library
                    if imp.startswith('.') or imp in sys.stdlib_module_names:
                        continue
                    
                    importlib.import_module(imp.split('.')[0])
                except ImportError:
                    missing_imports.append(imp)
            
            execution_time = time.time() - start_time
            
            if missing_imports:
                return ValidationResult(
                    module_name=module_path.stem,
                    test_type=TestType.IMPORT,
                    status=ValidationStatus.WARNING,
                    message=f"Missing imports: {', '.join(missing_imports)}",
                    execution_time=execution_time,
                    details={'missing_imports': missing_imports, 'total_imports': len(imports)}
                )
            else:
                return ValidationResult(
                    module_name=module_path.stem,
                    test_type=TestType.IMPORT,
                    status=ValidationStatus.PASSED,
                    message=f"All {len(imports)} imports validated",
                    execution_time=execution_time,
                    details={'total_imports': len(imports)}
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                module_name=module_path.stem,
                test_type=TestType.IMPORT,
                status=ValidationStatus.ERROR,
                message=f"Import test error: {e}",
                execution_time=execution_time
            )
    
    def _test_module_loading(self, module_path: Path) -> List[ValidationResult]:
        """Test module loading and class instantiation"""
        results = []
        start_time = time.time()
        
        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
            if spec is None or spec.loader is None:
                results.append(ValidationResult(
                    module_name=module_path.stem,
                    test_type=TestType.INSTANTIATION,
                    status=ValidationStatus.FAILED,
                    message="Could not create module spec",
                    execution_time=time.time() - start_time
                ))
                return results
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find classes in module
            classes = []
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    classes.append((name, obj))
            
            if classes:
                for class_name, class_obj in classes:
                    # Try to instantiate classes (with basic error handling)
                    try:
                        # Get constructor signature
                        sig = inspect.signature(class_obj.__init__)
                        params = list(sig.parameters.values())[1:]  # Skip 'self'
                        
                        # Try instantiation with minimal args
                        if all(p.default != inspect.Parameter.empty for p in params):
                            # All params have defaults
                            instance = class_obj()
                            results.append(ValidationResult(
                                module_name=module_path.stem,
                                test_type=TestType.INSTANTIATION,
                                status=ValidationStatus.PASSED,
                                message=f"Successfully instantiated {class_name}",
                                execution_time=time.time() - start_time,
                                details={'class_name': class_name, 'has_methods': len(inspect.getmembers(instance, inspect.ismethod))}
                            ))
                        else:
                            # Requires arguments - mark as warning
                            results.append(ValidationResult(
                                module_name=module_path.stem,
                                test_type=TestType.INSTANTIATION,
                                status=ValidationStatus.WARNING,
                                message=f"{class_name} requires constructor arguments",
                                execution_time=time.time() - start_time,
                                details={'class_name': class_name, 'required_params': len(params)}
                            ))
                    
                    except Exception as e:
                        results.append(ValidationResult(
                            module_name=module_path.stem,
                            test_type=TestType.INSTANTIATION,
                            status=ValidationStatus.FAILED,
                            message=f"Failed to instantiate {class_name}: {e}",
                            execution_time=time.time() - start_time,
                            details={'class_name': class_name, 'error': str(e)}
                        ))
            else:
                results.append(ValidationResult(
                    module_name=module_path.stem,
                    test_type=TestType.INSTANTIATION,
                    status=ValidationStatus.WARNING,
                    message="No classes found in module",
                    execution_time=time.time() - start_time
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                module_name=module_path.stem,
                test_type=TestType.INSTANTIATION,
                status=ValidationStatus.ERROR,
                message=f"Module loading error: {e}",
                execution_time=time.time() - start_time
            ))
        
        return results
    
    def _test_functionality(self, module_path: Path) -> List[ValidationResult]:
        """Test basic functionality of module components"""
        results = []
        start_time = time.time()
        
        # For now, return basic functionality check
        results.append(ValidationResult(
            module_name=module_path.stem,
            test_type=TestType.FUNCTIONALITY,
            status=ValidationStatus.WARNING,
            message="Functionality tests not yet implemented",
            execution_time=time.time() - start_time
        ))
        
        return results
    
    def _test_integration(self, module_path: Path) -> List[ValidationResult]:
        """Test integration with architecture framework"""
        results = []
        start_time = time.time()
        
        # Test if module follows architecture patterns
        try:
            # Check for common architecture patterns
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            patterns = {
                'dependency_injection': 'DependencyContainer' in content,
                'service_registry': 'ServiceRegistry' in content or 'get_service_registry' in content,
                'architecture_framework': 'ArchitecturalLayer' in content or 'get_architecture_framework' in content,
                'logging': 'logging.getLogger' in content,
                'typing': 'from typing import' in content or 'import typing' in content
            }
            
            pattern_score = sum(patterns.values())
            
            if pattern_score >= 3:
                status = ValidationStatus.PASSED
                message = f"Good integration patterns ({pattern_score}/5)"
            elif pattern_score >= 1:
                status = ValidationStatus.WARNING
                message = f"Basic integration patterns ({pattern_score}/5)"
            else:
                status = ValidationStatus.FAILED
                message = f"Poor integration patterns ({pattern_score}/5)"
            
            results.append(ValidationResult(
                module_name=module_path.stem,
                test_type=TestType.INTEGRATION,
                status=status,
                message=message,
                execution_time=time.time() - start_time,
                details=patterns
            ))
        
        except Exception as e:
            results.append(ValidationResult(
                module_name=module_path.stem,
                test_type=TestType.INTEGRATION,
                status=ValidationStatus.ERROR,
                message=f"Integration test error: {e}",
                execution_time=time.time() - start_time
            ))
        
        return results
    
    def _test_performance(self, module_path: Path) -> List[ValidationResult]:
        """Test module performance characteristics"""
        results = []
        start_time = time.time()
        
        try:
            # Measure import time
            import_start = time.time()
            spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            import_time = time.time() - import_start
            
            # Performance thresholds
            if import_time < 0.1:
                status = ValidationStatus.PASSED
                message = f"Fast import time: {import_time:.3f}s"
            elif import_time < 0.5:
                status = ValidationStatus.WARNING
                message = f"Moderate import time: {import_time:.3f}s"
            else:
                status = ValidationStatus.FAILED
                message = f"Slow import time: {import_time:.3f}s"
            
            results.append(ValidationResult(
                module_name=module_path.stem,
                test_type=TestType.PERFORMANCE,
                status=status,
                message=message,
                execution_time=time.time() - start_time,
                details={'import_time': import_time}
            ))
        
        except Exception as e:
            results.append(ValidationResult(
                module_name=module_path.stem,
                test_type=TestType.PERFORMANCE,
                status=ValidationStatus.ERROR,
                message=f"Performance test error: {e}",
                execution_time=time.time() - start_time
            ))
        
        return results
    
    def _count_lines(self, module_path: Path) -> int:
        """Count lines in module file"""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except Exception:
            return 0
    
    def _generate_validation_summary(self):
        """Generate comprehensive validation summary"""
        try:
            summary_path = self.test_results_dir / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'statistics': self.stats,
                'configuration': self.validation_config,
                'modules_tested': len(self.validation_reports),
                'overall_success_rate': (self.stats['modules_passed'] / max(self.stats['modules_tested'], 1)) * 100,
                'test_success_rate': (self.stats['total_passed'] / max(self.stats['total_tests'], 1)) * 100,
                'module_reports': [
                    {
                        'module_name': report.module_name,
                        'overall_status': report.overall_status.value,
                        'success_rate': report.success_rate,
                        'total_tests': report.total_tests,
                        'validation_time': report.validation_time,
                        'file_size': report.metadata.get('file_size', 0),
                        'line_count': report.metadata.get('line_count', 0)
                    }
                    for report in self.validation_reports
                ]
            }
            
            import json
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Validation summary saved to: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate validation summary: {e}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics"""
        return {
            'modules_discovered': len(self.discovered_modules),
            'modules_tested': self.stats['modules_tested'],
            'modules_passed': self.stats['modules_passed'],
            'modules_failed': self.stats['modules_failed'],
            'modules_warning': self.stats['modules_warning'],
            'overall_success_rate': (self.stats['modules_passed'] / max(self.stats['modules_tested'], 1)) * 100,
            'total_tests': self.stats['total_tests'],
            'total_passed': self.stats['total_passed'],
            'total_failed': self.stats['total_failed'],
            'test_success_rate': (self.stats['total_passed'] / max(self.stats['total_tests'], 1)) * 100,
            'validation_uptime': (datetime.now() - self.stats['start_time']).total_seconds()
        }


# Global validator instance
_module_validator: Optional[IntelligenceModuleValidator] = None


def get_module_validator() -> IntelligenceModuleValidator:
    """Get global module validator instance"""
    global _module_validator
    if _module_validator is None:
        _module_validator = IntelligenceModuleValidator()
    return _module_validator


def validate_intelligence_modules() -> List[ModuleValidationReport]:
    """Validate all intelligence modules"""
    validator = get_module_validator()
    return validator.validate_all_modules()


if __name__ == "__main__":
    # Run validation if called directly
    logging.basicConfig(level=logging.INFO)
    reports = validate_intelligence_modules()
    
    print(f"\nValidation Complete:")
    print(f"Modules tested: {len(reports)}")
    
    for report in reports:
        print(f"  {report.module_name}: {report.overall_status.value} "
              f"({report.passed_tests}/{report.total_tests} tests passed)")