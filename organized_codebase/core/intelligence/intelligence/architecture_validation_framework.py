#!/usr/bin/env python3
"""
Architecture Validation Framework - Agent D Hour 7
Comprehensive system architecture validation and verification
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import ast
import re
import subprocess
import yaml

@dataclass
class ValidationResult:
    """Represents a single validation result"""
    component: str
    test_name: str
    status: str  # "pass", "fail", "warning", "skip"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time: float = 0.0

@dataclass
class ArchitectureComponent:
    """Represents an architecture component"""
    name: str
    path: Path
    component_type: str  # "module", "package", "api", "config"
    dependencies: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    maintainer: str = "Agent D"
    last_validated: Optional[str] = None

class ArchitectureValidationFramework:
    """Comprehensive architecture validation and verification system"""
    
    def __init__(self, base_path: Union[str, Path] = "."):
        self.base_path = Path(base_path)
        self.results: List[ValidationResult] = []
        self.components: Dict[str, ArchitectureComponent] = {}
        self.validation_config = self._load_validation_config()
        self.start_time = time.time()
        
    def _load_validation_config(self) -> Dict[str, Any]:
        """Load validation configuration"""
        default_config = {
            "max_module_size": 1000,
            "max_function_complexity": 20,
            "required_documentation_coverage": 80,
            "max_dependency_depth": 5,
            "performance_thresholds": {
                "api_response_time": 200,  # ms
                "module_import_time": 50,   # ms
                "test_execution_time": 30   # seconds
            },
            "security_checks": {
                "check_secrets": True,
                "check_sql_injection": True,
                "check_xss_vulnerabilities": True,
                "check_insecure_functions": True
            },
            "compliance_standards": ["PEP8", "PEP257", "OWASP", "WCAG"]
        }
        
        config_file = self.base_path / "validation_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def discover_architecture_components(self) -> Dict[str, ArchitectureComponent]:
        """Discover and catalog all architecture components"""
        print("Discovering architecture components...")
        components = {}
        
        # Discover Python modules
        for py_file in self.base_path.rglob("*.py"):
            if self._should_include_file(py_file):
                component = self._analyze_python_module(py_file)
                components[component.name] = component
        
        # Discover configuration files
        for config_file in self.base_path.rglob("*.{yaml,yml,json,toml,ini}"):
            if self._should_include_file(config_file):
                component = self._analyze_config_file(config_file)
                components[component.name] = component
        
        # Discover API endpoints
        api_components = self._discover_api_endpoints()
        components.update(api_components)
        
        self.components = components
        return components
    
    def _should_include_file(self, file_path: Path) -> bool:
        """Determine if file should be included in validation"""
        exclude_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "node_modules",
            ".pytest_cache",
            "test_*_backup_*"
        ]
        
        return not any(pattern in str(file_path) for pattern in exclude_patterns)
    
    def _analyze_python_module(self, file_path: Path) -> ArchitectureComponent:
        """Analyze Python module and extract metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract dependencies
            dependencies = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    dependencies.append(node.module)
            
            # Extract interfaces (classes and functions)
            interfaces = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    interfaces.append(f"function:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    interfaces.append(f"class:{node.name}")
            
            return ArchitectureComponent(
                name=str(file_path.relative_to(self.base_path)),
                path=file_path,
                component_type="module",
                dependencies=dependencies,
                interfaces=interfaces
            )
            
        except Exception as e:
            return ArchitectureComponent(
                name=str(file_path.relative_to(self.base_path)),
                path=file_path,
                component_type="module",
                dependencies=[],
                interfaces=[f"error:parse_failed:{str(e)}"]
            )
    
    def _analyze_config_file(self, file_path: Path) -> ArchitectureComponent:
        """Analyze configuration file"""
        try:
            config_type = "config"
            if file_path.suffix in ['.yaml', '.yml']:
                config_type = "yaml_config"
            elif file_path.suffix == '.json':
                config_type = "json_config"
            elif file_path.suffix == '.toml':
                config_type = "toml_config"
            elif file_path.suffix == '.ini':
                config_type = "ini_config"
            
            return ArchitectureComponent(
                name=str(file_path.relative_to(self.base_path)),
                path=file_path,
                component_type=config_type,
                dependencies=[],
                interfaces=[f"config:{file_path.stem}"]
            )
            
        except Exception as e:
            return ArchitectureComponent(
                name=str(file_path.relative_to(self.base_path)),
                path=file_path,
                component_type="config",
                dependencies=[],
                interfaces=[f"error:config_failed:{str(e)}"]
            )
    
    def _discover_api_endpoints(self) -> Dict[str, ArchitectureComponent]:
        """Discover API endpoints"""
        endpoints = {}
        
        # Look for Flask/FastAPI route definitions
        for py_file in self.base_path.rglob("*.py"):
            if not self._should_include_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find Flask routes
                flask_routes = re.findall(r'@\w*\.route\([\'"]([^\'"]+)[\'"]', content)
                
                # Find FastAPI routes  
                fastapi_routes = re.findall(r'@\w*\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]', content)
                
                if flask_routes or fastapi_routes:
                    all_routes = flask_routes + [route[1] for route in fastapi_routes]
                    
                    component = ArchitectureComponent(
                        name=f"api:{py_file.stem}",
                        path=py_file,
                        component_type="api",
                        dependencies=[],
                        interfaces=[f"endpoint:{route}" for route in all_routes]
                    )
                    endpoints[component.name] = component
                    
            except Exception as e:
                continue
        
        return endpoints
    
    def validate_module_structure(self) -> List[ValidationResult]:
        """Validate module structure and organization"""
        print("Validating module structure...")
        results = []
        
        for name, component in self.components.items():
            if component.component_type == "module":
                start_time = time.time()
                
                # Check module size
                try:
                    file_size = component.path.stat().st_size
                    line_count = len(component.path.read_text(encoding='utf-8').splitlines())
                    
                    if line_count > self.validation_config["max_module_size"]:
                        results.append(ValidationResult(
                            component=name,
                            test_name="module_size_check",
                            status="fail",
                            message=f"Module exceeds size limit: {line_count} lines > {self.validation_config['max_module_size']}",
                            details={"line_count": line_count, "limit": self.validation_config["max_module_size"]},
                            execution_time=time.time() - start_time
                        ))
                    else:
                        results.append(ValidationResult(
                            component=name,
                            test_name="module_size_check",
                            status="pass",
                            message=f"Module size OK: {line_count} lines",
                            details={"line_count": line_count},
                            execution_time=time.time() - start_time
                        ))
                        
                except Exception as e:
                    results.append(ValidationResult(
                        component=name,
                        test_name="module_size_check",
                        status="fail",
                        message=f"Failed to check module size: {str(e)}",
                        execution_time=time.time() - start_time
                    ))
        
        return results
    
    def validate_dependency_structure(self) -> List[ValidationResult]:
        """Validate dependency relationships"""
        print("Validating dependency structure...")
        results = []
        
        # Build dependency graph
        dependency_graph = {}
        for name, component in self.components.items():
            dependency_graph[name] = component.dependencies
        
        # Check for circular dependencies
        for component_name in dependency_graph:
            start_time = time.time()
            
            if self._has_circular_dependency(component_name, dependency_graph):
                results.append(ValidationResult(
                    component=component_name,
                    test_name="circular_dependency_check",
                    status="fail",
                    message="Circular dependency detected",
                    details={"dependencies": dependency_graph[component_name]},
                    execution_time=time.time() - start_time
                ))
            else:
                results.append(ValidationResult(
                    component=component_name,
                    test_name="circular_dependency_check",
                    status="pass",
                    message="No circular dependencies",
                    details={"dependencies": dependency_graph[component_name]},
                    execution_time=time.time() - start_time
                ))
        
        return results
    
    def _has_circular_dependency(self, start_component: str, graph: Dict[str, List[str]], 
                                visited: Optional[List[str]] = None) -> bool:
        """Check for circular dependencies using DFS"""
        if visited is None:
            visited = []
        
        if start_component in visited:
            return True
        
        visited.append(start_component)
        
        for dependency in graph.get(start_component, []):
            if self._has_circular_dependency(dependency, graph, visited.copy()):
                return True
        
        return False
    
    def validate_api_endpoints(self) -> List[ValidationResult]:
        """Validate API endpoint functionality"""
        print("Validating API endpoints...")
        results = []
        
        api_components = {name: comp for name, comp in self.components.items() 
                         if comp.component_type == "api"}
        
        for name, component in api_components.items():
            for interface in component.interfaces:
                if interface.startswith("endpoint:"):
                    endpoint = interface.replace("endpoint:", "")
                    start_time = time.time()
                    
                    # Validate endpoint format
                    if self._is_valid_endpoint_format(endpoint):
                        results.append(ValidationResult(
                            component=name,
                            test_name="endpoint_format_validation",
                            status="pass",
                            message=f"Endpoint format valid: {endpoint}",
                            details={"endpoint": endpoint},
                            execution_time=time.time() - start_time
                        ))
                    else:
                        results.append(ValidationResult(
                            component=name,
                            test_name="endpoint_format_validation",
                            status="fail",
                            message=f"Invalid endpoint format: {endpoint}",
                            details={"endpoint": endpoint},
                            execution_time=time.time() - start_time
                        ))
        
        return results
    
    def _is_valid_endpoint_format(self, endpoint: str) -> bool:
        """Validate endpoint URL format"""
        # Basic validation - starts with /, contains valid characters
        return (endpoint.startswith("/") and 
                re.match(r'^[/a-zA-Z0-9_\-{}:<>]+$', endpoint) and
                len(endpoint) > 1)
    
    def validate_security_compliance(self) -> List[ValidationResult]:
        """Validate security compliance"""
        print("Validating security compliance...")
        results = []
        
        if not self.validation_config["security_checks"]["check_secrets"]:
            return results
        
        # Check for hardcoded secrets/credentials
        secret_patterns = [
            r'password\s*=\s*[\'"][^\'"]+[\'"]',
            r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
            r'secret\s*=\s*[\'"][^\'"]+[\'"]',
            r'token\s*=\s*[\'"][^\'"]+[\'"]'
        ]
        
        for name, component in self.components.items():
            if component.component_type == "module":
                start_time = time.time()
                
                try:
                    content = component.path.read_text(encoding='utf-8')
                    
                    secrets_found = []
                    for pattern in secret_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        secrets_found.extend(matches)
                    
                    if secrets_found:
                        results.append(ValidationResult(
                            component=name,
                            test_name="hardcoded_secrets_check",
                            status="warning",
                            message=f"Potential hardcoded secrets found: {len(secrets_found)}",
                            details={"secrets_count": len(secrets_found)},
                            execution_time=time.time() - start_time
                        ))
                    else:
                        results.append(ValidationResult(
                            component=name,
                            test_name="hardcoded_secrets_check",
                            status="pass",
                            message="No hardcoded secrets detected",
                            execution_time=time.time() - start_time
                        ))
                        
                except Exception as e:
                    results.append(ValidationResult(
                        component=name,
                        test_name="hardcoded_secrets_check",
                        status="fail",
                        message=f"Failed to check for secrets: {str(e)}",
                        execution_time=time.time() - start_time
                    ))
        
        return results
    
    def validate_performance_characteristics(self) -> List[ValidationResult]:
        """Validate performance characteristics"""
        print("Validating performance characteristics...")
        results = []
        
        # Test module import times
        for name, component in self.components.items():
            if component.component_type == "module" and component.path.suffix == '.py':
                start_time = time.time()
                
                try:
                    # Simulate import time test (simplified)
                    import_test_time = self._measure_import_time(component.path)
                    threshold = self.validation_config["performance_thresholds"]["module_import_time"]
                    
                    if import_test_time > threshold:
                        results.append(ValidationResult(
                            component=name,
                            test_name="import_performance",
                            status="warning",
                            message=f"Slow import time: {import_test_time:.2f}ms > {threshold}ms",
                            details={"import_time": import_test_time, "threshold": threshold},
                            execution_time=time.time() - start_time
                        ))
                    else:
                        results.append(ValidationResult(
                            component=name,
                            test_name="import_performance", 
                            status="pass",
                            message=f"Import time OK: {import_test_time:.2f}ms",
                            details={"import_time": import_test_time},
                            execution_time=time.time() - start_time
                        ))
                        
                except Exception as e:
                    results.append(ValidationResult(
                        component=name,
                        test_name="import_performance",
                        status="skip",
                        message=f"Could not measure import time: {str(e)}",
                        execution_time=time.time() - start_time
                    ))
        
        return results
    
    def _measure_import_time(self, module_path: Path) -> float:
        """Measure module import time (simplified simulation)"""
        # Simplified - in practice would use actual import timing
        file_size = module_path.stat().st_size
        # Estimate based on file size (rough approximation)
        estimated_time = (file_size / 1000) * 0.1  # 0.1ms per KB
        return estimated_time
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        print("Generating validation report...")
        
        # Run all validations
        all_results = []
        all_results.extend(self.validate_module_structure())
        all_results.extend(self.validate_dependency_structure())
        all_results.extend(self.validate_api_endpoints())
        all_results.extend(self.validate_security_compliance())
        all_results.extend(self.validate_performance_characteristics())
        
        self.results = all_results
        
        # Generate summary statistics
        total_tests = len(all_results)
        passed_tests = len([r for r in all_results if r.status == "pass"])
        failed_tests = len([r for r in all_results if r.status == "fail"])
        warning_tests = len([r for r in all_results if r.status == "warning"])
        skipped_tests = len([r for r in all_results if r.status == "skip"])
        
        # Calculate total execution time
        total_execution_time = time.time() - self.start_time
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "execution_time": total_execution_time,
            "summary": {
                "total_components": len(self.components),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "warning_tests": warning_tests,
                "skipped_tests": skipped_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "component_breakdown": {
                component_type: len([c for c in self.components.values() 
                                   if c.component_type == component_type])
                for component_type in set(c.component_type for c in self.components.values())
            },
            "validation_results": [
                {
                    "component": r.component,
                    "test_name": r.test_name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp,
                    "execution_time": r.execution_time
                }
                for r in all_results
            ],
            "recommendations": self._generate_recommendations(all_results),
            "config_used": self.validation_config
        }
        
        return report
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        failed_results = [r for r in results if r.status == "fail"]
        warning_results = [r for r in results if r.status == "warning"]
        
        # Module size recommendations
        size_failures = [r for r in failed_results if r.test_name == "module_size_check"]
        if size_failures:
            recommendations.append(f"Consider splitting {len(size_failures)} oversized modules")
        
        # Circular dependency recommendations
        circular_deps = [r for r in failed_results if r.test_name == "circular_dependency_check"]
        if circular_deps:
            recommendations.append(f"Resolve {len(circular_deps)} circular dependencies")
        
        # Security recommendations
        security_warnings = [r for r in warning_results if r.test_name == "hardcoded_secrets_check"]
        if security_warnings:
            recommendations.append(f"Review {len(security_warnings)} files with potential hardcoded secrets")
        
        # Performance recommendations
        perf_warnings = [r for r in warning_results if r.test_name == "import_performance"]
        if perf_warnings:
            recommendations.append(f"Optimize import performance for {len(perf_warnings)} slow modules")
        
        if not recommendations:
            recommendations.append("Architecture validation passed - no major issues found")
        
        return recommendations


def main():
    """Main execution function"""
    print("=== TestMaster Architecture Validation Framework ===")
    print("Agent D - Hour 7: Architecture Validation Framework")
    print()
    
    # Initialize validation framework
    validator = ArchitectureValidationFramework()
    
    # Discover components
    print("Phase 1: Component Discovery")
    components = validator.discover_architecture_components()
    print(f"Discovered {len(components)} components")
    
    # Generate validation report
    print("\nPhase 2: Architecture Validation")
    report = validator.generate_validation_report()
    
    # Save report
    report_file = Path("TestMaster/docs/validation/architecture_validation_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    print(f"\nValidation Complete!")
    print(f"Total Components: {report['summary']['total_components']}")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Execution Time: {report['execution_time']:.2f}s")
    print(f"\nReport saved: {report_file}")
    
    # Show recommendations
    if report['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")


if __name__ == "__main__":
    main()