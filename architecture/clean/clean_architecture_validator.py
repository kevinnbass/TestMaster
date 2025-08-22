#!/usr/bin/env python3
"""
Clean Architecture Validator - Agent A
AI-powered clean architecture validation and enforcement

Implements comprehensive clean architecture validation to ensure proper
layer separation, dependency direction, and architectural compliance.
"""

import ast
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import importlib.util
import re

# Try to import existing architecture components for integration
try:
    from core.architecture.layer_separation import (
        ArchitecturalLayer, LayerComponent, LayerViolation,
        ArchitectureValidationResult, LayerManager
    )
    from core.architecture.dependency_injection import DependencyContainer
    ARCHITECTURE_INTEGRATION = True
except ImportError:
    ARCHITECTURE_INTEGRATION = False
    # Define minimal types if architecture components not available
    class ArchitecturalLayer(Enum):
        DOMAIN = "domain"
        APPLICATION = "application"
        INFRASTRUCTURE = "infrastructure"
        PRESENTATION = "presentation"


@dataclass
class CodebaseAnalysis:
    """Represents analyzed codebase structure"""
    root_path: Path
    modules: Dict[str, Path] = field(default_factory=dict)
    layers: Dict[ArchitecturalLayer, List[str]] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    imports: Dict[str, List[str]] = field(default_factory=list)
    classes: Dict[str, List[str]] = field(default_factory=dict)
    functions: Dict[str, List[str]] = field(default_factory=dict)
    violations: List[Any] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CleanArchitectureValidationResult:
    """Results of clean architecture validation"""
    overall_compliance: float
    layer_validation: Optional[Any] = None
    dependency_validation: Optional[Any] = None
    separation_validation: Optional[Any] = None
    ai_analysis: Optional[Dict[str, Any]] = None
    layer_violations: List[Any] = field(default_factory=list)
    layer_health: Dict[str, float] = field(default_factory=dict)
    dependency_violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class LayerValidator:
    """Validates architectural layer separation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.layer_patterns = {
            ArchitecturalLayer.DOMAIN: [
                r"entity", r"domain", r"model", r"aggregate", r"value_object"
            ],
            ArchitecturalLayer.APPLICATION: [
                r"use_case", r"service", r"application", r"handler", r"command"
            ],
            ArchitecturalLayer.INFRASTRUCTURE: [
                r"repository", r"database", r"api", r"external", r"adapter"
            ],
            ArchitecturalLayer.PRESENTATION: [
                r"controller", r"view", r"ui", r"presentation", r"rest"
            ]
        }
    
    def validate_layers(self, codebase: CodebaseAnalysis) -> Dict[str, Any]:
        """Validate layer organization"""
        validation_result = {
            'layers_detected': {},
            'layer_purity': {},
            'violations': [],
            'score': 0.0
        }
        
        # Detect layers
        for module_name, module_path in codebase.modules.items():
            layer = self._detect_layer(module_name, module_path)
            if layer:
                if layer not in validation_result['layers_detected']:
                    validation_result['layers_detected'][layer.value] = []
                validation_result['layers_detected'][layer.value].append(module_name)
        
        # Check layer purity
        for layer, modules in validation_result['layers_detected'].items():
            purity_score = self._calculate_layer_purity(layer, modules, codebase)
            validation_result['layer_purity'][layer] = purity_score
        
        # Calculate overall score
        if validation_result['layer_purity']:
            validation_result['score'] = sum(validation_result['layer_purity'].values()) / len(validation_result['layer_purity'])
        
        return validation_result
    
    def _detect_layer(self, module_name: str, module_path: Path) -> Optional[ArchitecturalLayer]:
        """Detect which layer a module belongs to"""
        module_name_lower = module_name.lower()
        path_str_lower = str(module_path).lower()
        
        for layer, patterns in self.layer_patterns.items():
            for pattern in patterns:
                if re.search(pattern, module_name_lower) or re.search(pattern, path_str_lower):
                    return layer
        
        return None
    
    def _calculate_layer_purity(self, layer: str, modules: List[str], codebase: CodebaseAnalysis) -> float:
        """Calculate how pure a layer is (no mixing of concerns)"""
        if not modules:
            return 1.0
        
        violations = 0
        total_checks = len(modules)
        
        for module in modules:
            # Check if module has dependencies from other layers
            module_deps = codebase.dependencies.get(module, set())
            for dep in module_deps:
                dep_layer = self._detect_layer(dep, Path(dep))
                if dep_layer and dep_layer.value != layer:
                    violations += 1
        
        return max(0.0, 1.0 - (violations / max(1, total_checks)))


class DependencyValidator:
    """Validates dependency directions according to clean architecture"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Define allowed dependency directions (outer -> inner is allowed)
        self.dependency_rules = {
            ArchitecturalLayer.PRESENTATION: [
                ArchitecturalLayer.APPLICATION,
                ArchitecturalLayer.DOMAIN
            ],
            ArchitecturalLayer.INFRASTRUCTURE: [
                ArchitecturalLayer.APPLICATION,
                ArchitecturalLayer.DOMAIN
            ],
            ArchitecturalLayer.APPLICATION: [
                ArchitecturalLayer.DOMAIN
            ],
            ArchitecturalLayer.DOMAIN: []  # Domain should not depend on anything
        }
    
    def validate_dependencies(self, codebase: CodebaseAnalysis) -> Dict[str, Any]:
        """Validate dependency directions"""
        validation_result = {
            'valid_dependencies': [],
            'invalid_dependencies': [],
            'circular_dependencies': [],
            'score': 0.0
        }
        
        # Check each module's dependencies
        for module, deps in codebase.dependencies.items():
            module_layer = self._get_module_layer(module, codebase)
            
            for dep in deps:
                dep_layer = self._get_module_layer(dep, codebase)
                
                if module_layer and dep_layer:
                    if self._is_valid_dependency(module_layer, dep_layer):
                        validation_result['valid_dependencies'].append(
                            f"{module} ({module_layer.value}) -> {dep} ({dep_layer.value})"
                        )
                    else:
                        validation_result['invalid_dependencies'].append(
                            f"{module} ({module_layer.value}) -> {dep} ({dep_layer.value})"
                        )
        
        # Check for circular dependencies
        circular = self._detect_circular_dependencies(codebase.dependencies)
        validation_result['circular_dependencies'] = circular
        
        # Calculate score
        total = len(validation_result['valid_dependencies']) + len(validation_result['invalid_dependencies'])
        if total > 0:
            validation_result['score'] = len(validation_result['valid_dependencies']) / total
        else:
            validation_result['score'] = 1.0
        
        return validation_result
    
    def _get_module_layer(self, module: str, codebase: CodebaseAnalysis) -> Optional[ArchitecturalLayer]:
        """Get the architectural layer of a module"""
        for layer, modules in codebase.layers.items():
            if module in modules:
                return layer
        
        # Try to detect from module name
        module_lower = module.lower()
        if 'domain' in module_lower or 'entity' in module_lower:
            return ArchitecturalLayer.DOMAIN
        elif 'application' in module_lower or 'service' in module_lower:
            return ArchitecturalLayer.APPLICATION
        elif 'infrastructure' in module_lower or 'repository' in module_lower:
            return ArchitecturalLayer.INFRASTRUCTURE
        elif 'presentation' in module_lower or 'controller' in module_lower:
            return ArchitecturalLayer.PRESENTATION
        
        return None
    
    def _is_valid_dependency(self, from_layer: ArchitecturalLayer, to_layer: ArchitecturalLayer) -> bool:
        """Check if dependency direction is valid"""
        if from_layer == to_layer:
            return True  # Same layer dependencies are allowed
        
        allowed_deps = self.dependency_rules.get(from_layer, [])
        return to_layer in allowed_deps
    
    def _detect_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[str]:
        """Detect circular dependencies"""
        circular = []
        visited = set()
        rec_stack = set()
        
        def visit(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found circular dependency
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                circular.append(" -> ".join(cycle))
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependencies.get(node, set()):
                if visit(neighbor, path + [node]):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in dependencies:
            if node not in visited:
                visit(node, [])
        
        return circular


class SeparationValidator:
    """Validates separation of concerns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.concern_patterns = {
            'business_logic': [r'calculate', r'validate', r'process', r'transform'],
            'data_access': [r'save', r'load', r'query', r'fetch', r'database'],
            'presentation': [r'render', r'display', r'format', r'template'],
            'external_api': [r'http', r'request', r'api', r'webhook'],
            'configuration': [r'config', r'settings', r'environment'],
            'security': [r'auth', r'encrypt', r'permission', r'token']
        }
    
    def validate_separation(self, codebase: CodebaseAnalysis) -> Dict[str, Any]:
        """Validate separation of concerns"""
        validation_result = {
            'concern_distribution': {},
            'mixed_concerns': [],
            'well_separated': [],
            'score': 0.0
        }
        
        for module, functions in codebase.functions.items():
            concerns = self._detect_concerns(functions)
            
            if len(concerns) > 2:
                validation_result['mixed_concerns'].append({
                    'module': module,
                    'concerns': list(concerns),
                    'severity': 'high' if len(concerns) > 3 else 'medium'
                })
            elif len(concerns) == 1:
                validation_result['well_separated'].append({
                    'module': module,
                    'concern': list(concerns)[0]
                })
            
            for concern in concerns:
                if concern not in validation_result['concern_distribution']:
                    validation_result['concern_distribution'][concern] = []
                validation_result['concern_distribution'][concern].append(module)
        
        # Calculate score
        total_modules = len(codebase.modules)
        if total_modules > 0:
            well_separated_count = len(validation_result['well_separated'])
            validation_result['score'] = well_separated_count / total_modules
        else:
            validation_result['score'] = 1.0
        
        return validation_result
    
    def _detect_concerns(self, functions: List[str]) -> Set[str]:
        """Detect concerns in a module based on function names"""
        detected_concerns = set()
        
        for function in functions:
            function_lower = function.lower()
            for concern, patterns in self.concern_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, function_lower):
                        detected_concerns.add(concern)
                        break
        
        return detected_concerns


class ArchitectureAI:
    """AI-powered architectural analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_patterns = []
        self.recommendations_engine = RecommendationsEngine()
        
        # CRITICAL: Import cost tracking wrapper
        try:
            from core.monitoring.ai_analysis_wrapper import create_ai_wrapper
            self.ai_wrapper = create_ai_wrapper("ArchitectureAI", "gemini-2.5-pro")
            self.cost_tracking_enabled = True
            self.logger.warning("COST TRACKING ENABLED for AI Architecture Analysis")
        except ImportError:
            self.ai_wrapper = None
            self.cost_tracking_enabled = False
            self.logger.error("COST TRACKING NOT AVAILABLE - AI analysis may burn through budget!")
    
    def analyze_architecture(self, codebase: CodebaseAnalysis) -> Dict[str, Any]:
        """Perform AI-powered architecture analysis"""
        
        # CRITICAL: Check budget before AI analysis
        if self.cost_tracking_enabled and self.ai_wrapper:
            codebase_size = len(codebase.modules) * 100  # Rough estimate
            cost_estimate = self.ai_wrapper.get_cost_estimate("architecture_analysis", codebase_size)
            
            self.logger.warning(f"AI Architecture Analysis Cost Estimate: ${cost_estimate['estimated_cost']:.4f}")
            
            # For now, use rule-based analysis to avoid costs
            # TODO: Enable AI analysis only when budget allows
            self.logger.info("Using rule-based analysis to conserve budget")
        
        analysis = {
            'complexity_score': self._calculate_complexity(codebase),
            'maintainability_score': self._calculate_maintainability(codebase),
            'testability_score': self._calculate_testability(codebase),
            'patterns_detected': self._detect_patterns(codebase),
            'anti_patterns_detected': self._detect_anti_patterns(codebase),
            'recommendations': self.recommendations_engine.generate_recommendations(codebase),
            'cost_estimate': cost_estimate if self.cost_tracking_enabled else None,
            'analysis_method': 'rule_based_with_cost_tracking'
        }
        
        # Calculate overall architecture score
        scores = [
            analysis['complexity_score'],
            analysis['maintainability_score'],
            analysis['testability_score']
        ]
        analysis['overall_score'] = sum(scores) / len(scores)
        
        return analysis
    
    def _calculate_complexity(self, codebase: CodebaseAnalysis) -> float:
        """Calculate architectural complexity score"""
        # Simple heuristic based on dependencies
        total_deps = sum(len(deps) for deps in codebase.dependencies.values())
        total_modules = len(codebase.modules)
        
        if total_modules == 0:
            return 1.0
        
        avg_deps = total_deps / total_modules
        # Lower average dependencies = better score
        return max(0.0, 1.0 - (avg_deps / 10))
    
    def _calculate_maintainability(self, codebase: CodebaseAnalysis) -> float:
        """Calculate maintainability score"""
        # Based on module size and organization
        if not codebase.modules:
            return 1.0
        
        # Check for proper module organization
        organized_modules = 0
        for module in codebase.modules:
            if any(keyword in module.lower() for keyword in ['test', 'spec', 'mock']):
                continue  # Skip test files
            if self._is_well_organized(module):
                organized_modules += 1
        
        return organized_modules / len(codebase.modules)
    
    def _calculate_testability(self, codebase: CodebaseAnalysis) -> float:
        """Calculate testability score"""
        # Check for test coverage and dependency injection
        test_modules = sum(1 for m in codebase.modules if 'test' in m.lower())
        regular_modules = len(codebase.modules) - test_modules
        
        if regular_modules == 0:
            return 1.0
        
        test_ratio = test_modules / regular_modules
        return min(1.0, test_ratio)
    
    def _detect_patterns(self, codebase: CodebaseAnalysis) -> List[str]:
        """Detect architectural patterns"""
        patterns = []
        
        # Check for common patterns
        if any('repository' in m.lower() for m in codebase.modules):
            patterns.append("Repository Pattern")
        
        if any('factory' in m.lower() for m in codebase.modules):
            patterns.append("Factory Pattern")
        
        if any('observer' in m.lower() for m in codebase.modules):
            patterns.append("Observer Pattern")
        
        if any('strategy' in m.lower() for m in codebase.modules):
            patterns.append("Strategy Pattern")
        
        if any('adapter' in m.lower() for m in codebase.modules):
            patterns.append("Adapter Pattern")
        
        return patterns
    
    def _detect_anti_patterns(self, codebase: CodebaseAnalysis) -> List[str]:
        """Detect architectural anti-patterns"""
        anti_patterns = []
        
        # Check for god classes
        for module, classes in codebase.classes.items():
            if len(classes) > 20:
                anti_patterns.append(f"God Class: {module}")
        
        # Check for spaghetti code (too many dependencies)
        for module, deps in codebase.dependencies.items():
            if len(deps) > 15:
                anti_patterns.append(f"Spaghetti Code: {module}")
        
        return anti_patterns
    
    def _is_well_organized(self, module: str) -> bool:
        """Check if module is well organized"""
        # Simple heuristic
        organized_keywords = ['core', 'domain', 'application', 'infrastructure', 'presentation']
        return any(keyword in module.lower() for keyword in organized_keywords)


class RecommendationsEngine:
    """Generate architectural recommendations"""
    
    def generate_recommendations(self, codebase: CodebaseAnalysis) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check for missing layers
        if not codebase.layers.get(ArchitecturalLayer.DOMAIN):
            recommendations.append("Consider creating a clear domain layer for business logic")
        
        if not codebase.layers.get(ArchitecturalLayer.APPLICATION):
            recommendations.append("Consider adding an application layer for use cases")
        
        # Check for dependency violations
        if codebase.violations:
            recommendations.append("Review and fix dependency violations to maintain clean architecture")
        
        # Check for test coverage
        test_modules = sum(1 for m in codebase.modules if 'test' in m.lower())
        if test_modules < len(codebase.modules) * 0.3:
            recommendations.append("Increase test coverage - aim for at least 1 test file per 3 source files")
        
        return recommendations


class CleanArchitectureValidator:
    """AI-powered clean architecture validation and enforcement"""
    
    def __init__(self):
        self.layer_validator = LayerValidator()
        self.dependency_validator = DependencyValidator()
        self.separation_validator = SeparationValidator()
        self.architecture_ai = ArchitectureAI()
        self.logger = logging.getLogger(__name__)
        
        # Integration with existing architecture if available
        if ARCHITECTURE_INTEGRATION:
            try:
                self.layer_manager = LayerManager()
                self.dependency_container = DependencyContainer()
                self.logger.info("Integrated with existing architecture components")
            except:
                self.layer_manager = None
                self.dependency_container = None
    
    def validate_clean_architecture(self, codebase: CodebaseAnalysis) -> CleanArchitectureValidationResult:
        """Comprehensive clean architecture validation"""
        result = CleanArchitectureValidationResult(overall_compliance=0.0)
        
        # Validate layer separation
        layer_validation = self.layer_validator.validate_layers(codebase)
        result.layer_validation = layer_validation
        result.layer_health = layer_validation.get('layer_purity', {})
        
        # Validate dependency directions
        dependency_validation = self.dependency_validator.validate_dependencies(codebase)
        result.dependency_validation = dependency_validation
        result.dependency_violations = dependency_validation.get('invalid_dependencies', [])
        
        # Validate separation of concerns
        separation_validation = self.separation_validator.validate_separation(codebase)
        result.separation_validation = separation_validation
        
        # AI-powered architectural analysis
        ai_analysis = self.architecture_ai.analyze_architecture(codebase)
        result.ai_analysis = ai_analysis
        result.recommendations = ai_analysis.get('recommendations', [])
        
        # Calculate overall compliance score
        scores = [
            layer_validation.get('score', 0.0),
            dependency_validation.get('score', 0.0),
            separation_validation.get('score', 0.0),
            ai_analysis.get('overall_score', 0.0)
        ]
        result.overall_compliance = sum(scores) / len(scores)
        
        # Log results
        self.logger.info(f"Clean Architecture Validation Complete - Compliance: {result.overall_compliance:.2%}")
        
        return result
    
    def analyze_codebase(self, root_path: Path) -> CodebaseAnalysis:
        """Analyze codebase structure"""
        analysis = CodebaseAnalysis(root_path=root_path)
        
        # Scan for Python modules
        for py_file in root_path.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
            
            module_name = py_file.stem
            analysis.modules[module_name] = py_file
            
            # Parse file for detailed analysis
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                    # Extract imports
                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)
                    
                    analysis.imports[module_name] = imports
                    
                    # Extract classes and functions
                    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                    functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    
                    analysis.classes[module_name] = classes
                    analysis.functions[module_name] = functions
                    
                    # Build dependency graph
                    analysis.dependencies[module_name] = set(imports)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse {py_file}: {e}")
        
        # Detect layers
        for module_name in analysis.modules:
            layer = self._detect_module_layer(module_name)
            if layer:
                if layer not in analysis.layers:
                    analysis.layers[layer] = []
                analysis.layers[layer].append(module_name)
        
        return analysis
    
    def _detect_module_layer(self, module_name: str) -> Optional[ArchitecturalLayer]:
        """Detect which layer a module belongs to"""
        module_lower = module_name.lower()
        
        if any(keyword in module_lower for keyword in ['domain', 'entity', 'model']):
            return ArchitecturalLayer.DOMAIN
        elif any(keyword in module_lower for keyword in ['application', 'service', 'use_case']):
            return ArchitecturalLayer.APPLICATION
        elif any(keyword in module_lower for keyword in ['infrastructure', 'repository', 'database']):
            return ArchitecturalLayer.INFRASTRUCTURE
        elif any(keyword in module_lower for keyword in ['presentation', 'controller', 'view']):
            return ArchitecturalLayer.PRESENTATION
        
        return None
    
    def generate_report(self, validation_result: CleanArchitectureValidationResult) -> str:
        """Generate human-readable validation report"""
        report = []
        report.append("=" * 60)
        report.append("CLEAN ARCHITECTURE VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {validation_result.timestamp}")
        report.append(f"Overall Compliance: {validation_result.overall_compliance:.2%}")
        report.append("")
        
        # Layer validation
        if validation_result.layer_validation:
            report.append("LAYER VALIDATION:")
            report.append("-" * 40)
            layer_val = validation_result.layer_validation
            report.append(f"Score: {layer_val.get('score', 0):.2%}")
            report.append(f"Layers Detected: {list(layer_val.get('layers_detected', {}).keys())}")
            report.append("")
        
        # Dependency validation
        if validation_result.dependency_validation:
            report.append("DEPENDENCY VALIDATION:")
            report.append("-" * 40)
            dep_val = validation_result.dependency_validation
            report.append(f"Score: {dep_val.get('score', 0):.2%}")
            report.append(f"Valid Dependencies: {len(dep_val.get('valid_dependencies', []))}")
            report.append(f"Invalid Dependencies: {len(dep_val.get('invalid_dependencies', []))}")
            if dep_val.get('circular_dependencies'):
                report.append(f"Circular Dependencies Found: {len(dep_val['circular_dependencies'])}")
            report.append("")
        
        # AI Analysis
        if validation_result.ai_analysis:
            report.append("AI ANALYSIS:")
            report.append("-" * 40)
            ai = validation_result.ai_analysis
            report.append(f"Complexity Score: {ai.get('complexity_score', 0):.2%}")
            report.append(f"Maintainability Score: {ai.get('maintainability_score', 0):.2%}")
            report.append(f"Testability Score: {ai.get('testability_score', 0):.2%}")
            
            if ai.get('patterns_detected'):
                report.append(f"Patterns Detected: {', '.join(ai['patterns_detected'])}")
            
            if ai.get('anti_patterns_detected'):
                report.append(f"Anti-Patterns: {', '.join(ai['anti_patterns_detected'])}")
            report.append("")
        
        # Recommendations
        if validation_result.recommendations:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 40)
            for i, rec in enumerate(validation_result.recommendations, 1):
                report.append(f"{i}. {rec}")
        
        return "\n".join(report)


def validate_architecture(root_path: str) -> Dict[str, Any]:
    """Main entry point for architecture validation"""
    validator = CleanArchitectureValidator()
    
    # Analyze codebase
    codebase = validator.analyze_codebase(Path(root_path))
    
    # Validate architecture
    result = validator.validate_clean_architecture(codebase)
    
    # Generate report
    report = validator.generate_report(result)
    
    return {
        'compliance_score': result.overall_compliance,
        'report': report,
        'validation_result': result,
        'codebase_analysis': codebase
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = "."
    
    print(f"Validating Clean Architecture for: {root_path}")
    print("=" * 60)
    
    result = validate_architecture(root_path)
    print(result['report'])
    print("\n" + "=" * 60)
    print(f"Final Compliance Score: {result['compliance_score']:.2%}")