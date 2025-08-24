#!/usr/bin/env python3
"""
TestMaster Interface Analysis - Phase 2: Hours 36-40
Agent B - Documentation & Modularization Excellence

Comprehensive interface analysis and contract documentation system.
Analyzes module interfaces, documents contracts, calculates interface stability,
and identifies interface violations.
"""

import ast
import logging
import os
import json
import re
import inspect
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from datetime import datetime
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InterfaceSignature:
    """Detailed function/method signature analysis."""
    name: str
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    is_public: bool
    is_async: bool
    decorators: List[str]
    complexity_score: float
    
@dataclass 
class ClassInterface:
    """Comprehensive class interface definition."""
    name: str
    methods: List[InterfaceSignature]
    properties: List[str]
    inheritance: List[str]
    abstract_methods: List[str]
    public_attributes: List[str]
    docstring: Optional[str]
    is_abstract: bool
    interface_stability: float

@dataclass
class ModuleInterface:
    """Complete module interface specification."""
    module_id: str
    module_path: Path
    functions: List[InterfaceSignature]
    classes: List[ClassInterface]
    constants: List[str]
    imports: List[str]
    exports: List[str]
    docstring: Optional[str]
    api_version: Optional[str]
    deprecation_warnings: List[str]
    interface_complexity: float
    contract_violations: List[str]
    stability_score: float

@dataclass
class InterfaceContract:
    """Interface contract definition and validation."""
    contract_id: str
    expected_signature: str
    actual_signature: str
    is_compatible: bool
    breaking_changes: List[str]
    deprecation_status: str
    version_introduced: Optional[str]
    version_deprecated: Optional[str]

@dataclass
class InterfaceViolation:
    """Interface violation detection and reporting."""
    violation_type: str
    severity: str  # critical, high, medium, low
    module_id: str
    function_name: str
    description: str
    recommendation: str
    impact_assessment: str

class InterfaceAnalyzer:
    """Comprehensive interface analysis and documentation system."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.module_interfaces: Dict[str, ModuleInterface] = {}
        self.interface_contracts: Dict[str, InterfaceContract] = {}
        self.interface_violations: List[InterfaceViolation] = []
        self.interface_metrics = {
            'total_modules_analyzed': 0,
            'total_public_functions': 0,
            'total_public_classes': 0,
            'interface_violations_found': 0,
            'contract_violations_found': 0,
            'average_interface_stability': 0.0,
            'interface_complexity_distribution': {},
            'documentation_coverage': 0.0
        }
        
    def analyze_all_interfaces(self) -> Dict[str, Any]:
        """Perform comprehensive interface analysis across the framework."""
        logger.info("🔍 Starting comprehensive interface analysis...")
        
        # Critical modules for detailed analysis
        critical_modules = [
            "core/intelligence/__init__.py",
            "core/intelligence/testing/__init__.py", 
            "core/intelligence/api/__init__.py",
            "core/intelligence/analytics/__init__.py",
            "testmaster_orchestrator.py",
            "intelligent_test_builder.py",
            "enhanced_self_healing_verifier.py",
            "agentic_test_monitor.py",
            "parallel_converter.py",
            "config/__init__.py"
        ]
        
        for module_path in critical_modules:
            try:
                full_path = self.base_path / module_path
                if full_path.exists():
                    logger.info(f"Analyzing interface: {module_path}")
                    interface = self._analyze_module_interface(full_path, module_path)
                    self.module_interfaces[module_path] = interface
                    self._detect_interface_violations(interface)
                else:
                    logger.warning(f"Module not found: {module_path}")
            except Exception as e:
                logger.error(f"Error analyzing interface {module_path}: {e}")
        
        self._calculate_interface_metrics()
        self._analyze_interface_contracts()
        self._generate_interface_stability_scores()
        
        return self._compile_interface_analysis_results()
    
    def _analyze_module_interface(self, file_path: Path, module_id: str) -> ModuleInterface:
        """Analyze a single module's interface comprehensively."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Extract module-level information
            module_docstring = ast.get_docstring(tree)
            functions = self._extract_functions(tree)
            classes = self._extract_classes(tree)
            constants = self._extract_constants(tree)
            imports = self._extract_imports(tree)
            exports = self._extract_exports(tree, content)
            
            # Calculate interface complexity
            interface_complexity = self._calculate_interface_complexity(functions, classes)
            
            # Detect contract violations
            contract_violations = self._detect_contract_violations(functions, classes)
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(functions, classes)
            
            # Check for deprecation warnings
            deprecation_warnings = self._find_deprecation_warnings(content)
            
            # Extract API version if present
            api_version = self._extract_api_version(content)
            
            return ModuleInterface(
                module_id=module_id,
                module_path=file_path,
                functions=functions,
                classes=classes,
                constants=constants,
                imports=imports,
                exports=exports,
                docstring=module_docstring,
                api_version=api_version,
                deprecation_warnings=deprecation_warnings,
                interface_complexity=interface_complexity,
                contract_violations=contract_violations,
                stability_score=stability_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing module interface {file_path}: {e}")
            return self._create_empty_interface(module_id, file_path)
    
    def _extract_functions(self, tree: ast.AST) -> List[InterfaceSignature]:
        """Extract function signatures and analyze their interfaces."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Only analyze module-level functions (not methods)
                if self._is_module_level_function(node, tree):
                    signature = self._analyze_function_signature(node)
                    functions.append(signature)
        
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[ClassInterface]:
        """Extract class interfaces and analyze their contracts."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_interface = self._analyze_class_interface(node)
                classes.append(class_interface)
        
        return classes
    
    def _analyze_function_signature(self, node: ast.FunctionDef) -> InterfaceSignature:
        """Analyze a function's signature in detail."""
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_str = arg.arg
            if arg.annotation:
                param_str += f": {ast.unparse(arg.annotation)}"
            parameters.append(param_str)
        
        # Add default arguments
        defaults_offset = len(node.args.args) - len(node.args.defaults)
        for i, default in enumerate(node.args.defaults):
            param_idx = defaults_offset + i
            if param_idx < len(parameters):
                parameters[param_idx] += f" = {ast.unparse(default)}"
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        # Extract decorators
        decorators = [ast.unparse(dec) for dec in node.decorator_list]
        
        # Check if function is async
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # Check if function is public (not starting with _)
        is_public = not node.name.startswith('_')
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Calculate complexity score
        complexity_score = self._calculate_function_complexity(node)
        
        return InterfaceSignature(
            name=node.name,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            is_public=is_public,
            is_async=is_async,
            decorators=decorators,
            complexity_score=complexity_score
        )
    
    def _analyze_class_interface(self, node: ast.ClassDef) -> ClassInterface:
        """Analyze a class interface comprehensively."""
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_signature = self._analyze_function_signature(item)
                methods.append(method_signature)
        
        # Extract properties
        properties = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if any(isinstance(dec, ast.Name) and dec.id == 'property' 
                       for dec in item.decorator_list):
                    properties.append(item.name)
        
        # Extract inheritance
        inheritance = []
        for base in node.bases:
            inheritance.append(ast.unparse(base))
        
        # Find abstract methods
        abstract_methods = []
        for method in methods:
            if 'abstractmethod' in method.decorators:
                abstract_methods.append(method.name)
        
        # Extract public attributes (from __init__ if present)
        public_attributes = self._extract_public_attributes(node)
        
        # Get class docstring
        docstring = ast.get_docstring(node)
        
        # Check if class is abstract
        is_abstract = len(abstract_methods) > 0 or 'ABC' in inheritance
        
        # Calculate interface stability
        interface_stability = self._calculate_class_stability(methods, properties)
        
        return ClassInterface(
            name=node.name,
            methods=methods,
            properties=properties,
            inheritance=inheritance,
            abstract_methods=abstract_methods,
            public_attributes=public_attributes,
            docstring=docstring,
            is_abstract=is_abstract,
            interface_stability=interface_stability
        )
    
    def _extract_constants(self, tree: ast.AST) -> List[str]:
        """Extract module-level constants."""
        constants = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Constants are typically ALL_CAPS
                        if target.id.isupper() and target.id.replace('_', '').isalpha():
                            constants.append(target.id)
        
        return constants
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all import statements."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        return imports
    
    def _extract_exports(self, tree: ast.AST, content: str) -> List[str]:
        """Extract module exports (__all__ or public functions/classes)."""
        exports = []
        
        # First, try to find __all__
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Str):
                                    exports.append(elt.s)
                                elif isinstance(elt, ast.Constant):
                                    exports.append(str(elt.value))
        
        # If no __all__, find public functions and classes
        if not exports:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not node.name.startswith('_'):
                        exports.append(node.name)
        
        return exports
    
    def _calculate_interface_complexity(self, functions: List[InterfaceSignature], 
                                      classes: List[ClassInterface]) -> float:
        """Calculate overall interface complexity score."""
        if not functions and not classes:
            return 0.0
        
        total_complexity = 0.0
        total_items = 0
        
        # Function complexity
        for func in functions:
            if func.is_public:
                total_complexity += func.complexity_score
                total_items += 1
        
        # Class complexity
        for cls in classes:
            if not cls.name.startswith('_'):
                # Class complexity is average of public method complexities
                public_methods = [m for m in cls.methods if m.is_public]
                if public_methods:
                    class_complexity = sum(m.complexity_score for m in public_methods) / len(public_methods)
                    total_complexity += class_complexity
                    total_items += 1
        
        return total_complexity / total_items if total_items > 0 else 0.0
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> float:
        """Calculate function complexity based on various factors."""
        complexity = 1.0  # Base complexity
        
        # Parameter complexity
        param_count = len(node.args.args)
        complexity += param_count * 0.1
        
        # Control flow complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 0.5
            elif isinstance(child, ast.comprehension):
                complexity += 0.3
        
        # Nested function complexity
        nested_functions = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n != node]
        complexity += len(nested_functions) * 0.4
        
        return complexity
    
    def _calculate_class_stability(self, methods: List[InterfaceSignature], 
                                 properties: List[str]) -> float:
        """Calculate class interface stability score."""
        stability_factors = []
        
        # Method stability
        public_methods = [m for m in methods if m.is_public]
        if public_methods:
            # More public methods = potentially less stable
            method_stability = max(0.0, 1.0 - (len(public_methods) - 5) * 0.1)
            stability_factors.append(method_stability)
        
        # Property stability
        property_stability = max(0.0, 1.0 - len(properties) * 0.05)
        stability_factors.append(property_stability)
        
        # Documentation stability (methods with docstrings are more stable)
        documented_methods = sum(1 for m in public_methods if m.docstring)
        doc_stability = documented_methods / len(public_methods) if public_methods else 1.0
        stability_factors.append(doc_stability)
        
        return statistics.mean(stability_factors) if stability_factors else 0.5
    
    def _calculate_stability_score(self, functions: List[InterfaceSignature], 
                                 classes: List[ClassInterface]) -> float:
        """Calculate overall module interface stability."""
        stability_factors = []
        
        # Function stability
        public_functions = [f for f in functions if f.is_public]
        if public_functions:
            # Documentation coverage affects stability
            documented_funcs = sum(1 for f in public_functions if f.docstring)
            func_doc_stability = documented_funcs / len(public_functions)
            stability_factors.append(func_doc_stability)
            
            # Complexity affects stability (lower complexity = higher stability)
            avg_complexity = statistics.mean(f.complexity_score for f in public_functions)
            complexity_stability = max(0.0, 1.0 - (avg_complexity - 2.0) * 0.1)
            stability_factors.append(complexity_stability)
        
        # Class stability
        public_classes = [c for c in classes if not c.name.startswith('_')]
        if public_classes:
            class_stabilities = [c.interface_stability for c in public_classes]
            avg_class_stability = statistics.mean(class_stabilities)
            stability_factors.append(avg_class_stability)
        
        return statistics.mean(stability_factors) if stability_factors else 0.5
    
    def _detect_contract_violations(self, functions: List[InterfaceSignature], 
                                  classes: List[ClassInterface]) -> List[str]:
        """Detect interface contract violations."""
        violations = []
        
        # Check function contract violations
        for func in functions:
            if func.is_public:
                # Missing docstring
                if not func.docstring:
                    violations.append(f"Function '{func.name}' missing docstring")
                
                # Missing type hints
                if not func.return_type and func.name != '__init__':
                    violations.append(f"Function '{func.name}' missing return type hint")
                
                # Complex parameter list without documentation
                if len(func.parameters) > 5 and not func.docstring:
                    violations.append(f"Function '{func.name}' has complex parameters but no documentation")
        
        # Check class contract violations
        for cls in classes:
            if not cls.name.startswith('_'):
                # Missing class docstring
                if not cls.docstring:
                    violations.append(f"Class '{cls.name}' missing docstring")
                
                # Public methods without docstrings
                for method in cls.methods:
                    if method.is_public and not method.docstring and method.name != '__init__':
                        violations.append(f"Method '{cls.name}.{method.name}' missing docstring")
        
        return violations
    
    def _detect_interface_violations(self, interface: ModuleInterface):
        """Detect and record interface violations."""
        for violation_desc in interface.contract_violations:
            # Parse violation description to create structured violation
            if "missing docstring" in violation_desc.lower():
                severity = "medium"
                violation_type = "missing_documentation"
            elif "missing return type" in violation_desc.lower():
                severity = "low"
                violation_type = "missing_type_hint"
            elif "complex parameters" in violation_desc.lower():
                severity = "high"
                violation_type = "complex_interface"
            else:
                severity = "low"
                violation_type = "general_violation"
            
            violation = InterfaceViolation(
                violation_type=violation_type,
                severity=severity,
                module_id=interface.module_id,
                function_name="",  # Would need to parse from description
                description=violation_desc,
                recommendation=self._generate_violation_recommendation(violation_type),
                impact_assessment=self._assess_violation_impact(violation_type, severity)
            )
            
            self.interface_violations.append(violation)
    
    def _generate_violation_recommendation(self, violation_type: str) -> str:
        """Generate recommendation for fixing interface violation."""
        recommendations = {
            "missing_documentation": "Add comprehensive docstring with parameter descriptions and return value documentation",
            "missing_type_hint": "Add type hints using typing module for better interface clarity",
            "complex_interface": "Consider splitting complex function or add detailed parameter documentation",
            "general_violation": "Review interface design and follow established conventions"
        }
        return recommendations.get(violation_type, "Review and fix interface issue")
    
    def _assess_violation_impact(self, violation_type: str, severity: str) -> str:
        """Assess the impact of an interface violation."""
        impact_matrix = {
            ("missing_documentation", "high"): "Severely impacts maintainability and onboarding",
            ("missing_documentation", "medium"): "Reduces code understandability and maintainability",
            ("missing_type_hint", "medium"): "Affects IDE support and type checking",
            ("complex_interface", "high"): "Makes interface difficult to use and error-prone",
            ("general_violation", "low"): "Minor impact on code quality"
        }
        
        key = (violation_type, severity)
        return impact_matrix.get(key, "Impact assessment needed")
    
    def _find_deprecation_warnings(self, content: str) -> List[str]:
        """Find deprecation warnings in module content."""
        warnings = []
        deprecation_patterns = [
            r'warnings\.warn.*[Dd]eprecat',
            r'@deprecated',
            r'# DEPRECATED',
            r'# TODO: deprecate'
        ]
        
        for pattern in deprecation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            warnings.extend(matches)
        
        return warnings
    
    def _extract_api_version(self, content: str) -> Optional[str]:
        """Extract API version from module content."""
        version_patterns = [
            r'__version__\s*=\s*["\']([^"\']+)["\']',
            r'API_VERSION\s*=\s*["\']([^"\']+)["\']',
            r'VERSION\s*=\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_public_attributes(self, node: ast.ClassDef) -> List[str]:
        """Extract public attributes from class __init__ method."""
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for stmt in item.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute):
                                if (isinstance(target.value, ast.Name) and 
                                    target.value.id == 'self' and 
                                    not target.attr.startswith('_')):
                                    attributes.append(target.attr)
        
        return attributes
    
    def _is_module_level_function(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is at module level (not a method)."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return False
        return True
    
    def _calculate_interface_metrics(self):
        """Calculate comprehensive interface metrics."""
        self.interface_metrics['total_modules_analyzed'] = len(self.module_interfaces)
        
        total_public_functions = 0
        total_public_classes = 0
        total_documented_items = 0
        total_items = 0
        complexity_scores = []
        stability_scores = []
        
        for interface in self.module_interfaces.values():
            # Count public functions
            public_funcs = [f for f in interface.functions if f.is_public]
            total_public_functions += len(public_funcs)
            
            # Count public classes
            public_classes = [c for c in interface.classes if not c.name.startswith('_')]
            total_public_classes += len(public_classes)
            
            # Documentation coverage
            documented_funcs = sum(1 for f in public_funcs if f.docstring)
            documented_classes = sum(1 for c in public_classes if c.docstring)
            total_documented_items += documented_funcs + documented_classes
            total_items += len(public_funcs) + len(public_classes)
            
            # Complexity and stability
            complexity_scores.append(interface.interface_complexity)
            stability_scores.append(interface.stability_score)
        
        self.interface_metrics['total_public_functions'] = total_public_functions
        self.interface_metrics['total_public_classes'] = total_public_classes
        self.interface_metrics['interface_violations_found'] = len(self.interface_violations)
        self.interface_metrics['documentation_coverage'] = (
            total_documented_items / total_items if total_items > 0 else 0.0
        )
        
        if stability_scores:
            self.interface_metrics['average_interface_stability'] = statistics.mean(stability_scores)
        
        # Complexity distribution
        if complexity_scores:
            self.interface_metrics['interface_complexity_distribution'] = {
                'low': sum(1 for score in complexity_scores if score < 2.0),
                'medium': sum(1 for score in complexity_scores if 2.0 <= score < 4.0),
                'high': sum(1 for score in complexity_scores if score >= 4.0),
                'average': statistics.mean(complexity_scores)
            }
    
    def _analyze_interface_contracts(self):
        """Analyze interface contracts across modules."""
        # This would compare interfaces between versions or expected contracts
        # For now, we'll create basic contract validation
        
        for module_id, interface in self.module_interfaces.items():
            for func in interface.functions:
                if func.is_public:
                    contract_id = f"{module_id}::{func.name}"
                    
                    # Create basic contract
                    contract = InterfaceContract(
                        contract_id=contract_id,
                        expected_signature=self._format_function_signature(func),
                        actual_signature=self._format_function_signature(func),
                        is_compatible=True,
                        breaking_changes=[],
                        deprecation_status="active",
                        version_introduced=interface.api_version,
                        version_deprecated=None
                    )
                    
                    self.interface_contracts[contract_id] = contract
    
    def _format_function_signature(self, func: InterfaceSignature) -> str:
        """Format function signature for contract comparison."""
        params = ", ".join(func.parameters)
        return_part = f" -> {func.return_type}" if func.return_type else ""
        return f"{func.name}({params}){return_part}"
    
    def _generate_interface_stability_scores(self):
        """Generate interface stability scores for critical components."""
        stability_categories = {
            'critical': [],  # Core infrastructure
            'high': [],      # Important but stable
            'medium': [],    # Application level
            'low': []        # Utilities and helpers
        }
        
        for module_id, interface in self.module_interfaces.items():
            if 'core/intelligence' in module_id:
                stability_categories['critical'].append(interface.stability_score)
            elif any(x in module_id for x in ['config', 'api']):
                stability_categories['high'].append(interface.stability_score)
            elif any(x in module_id for x in ['orchestrator', 'builder']):
                stability_categories['medium'].append(interface.stability_score)
            else:
                stability_categories['low'].append(interface.stability_score)
        
        # Calculate average stability by category
        for category, scores in stability_categories.items():
            if scores:
                avg_stability = statistics.mean(scores)
                self.interface_metrics[f'{category}_interface_stability'] = avg_stability
    
    def _create_empty_interface(self, module_id: str, file_path: Path) -> ModuleInterface:
        """Create empty interface for failed analysis."""
        return ModuleInterface(
            module_id=module_id,
            module_path=file_path,
            functions=[],
            classes=[],
            constants=[],
            imports=[],
            exports=[],
            docstring=None,
            api_version=None,
            deprecation_warnings=[],
            interface_complexity=0.0,
            contract_violations=[],
            stability_score=0.0
        )
    
    def _compile_interface_analysis_results(self) -> Dict[str, Any]:
        """Compile comprehensive interface analysis results."""
        return {
            "analysis_metadata": {
                "analyzer": "Agent B - Interface Analysis",
                "phase": "Hours 36-40",
                "modules_analyzed": len(self.module_interfaces),
                "total_violations": len(self.interface_violations)
            },
            "module_interfaces": {
                module_id: {
                    "functions": len(interface.functions),
                    "public_functions": len([f for f in interface.functions if f.is_public]),
                    "classes": len(interface.classes),
                    "public_classes": len([c for c in interface.classes if not c.name.startswith('_')]),
                    "constants": len(interface.constants),
                    "exports": len(interface.exports),
                    "interface_complexity": interface.interface_complexity,
                    "stability_score": interface.stability_score,
                    "contract_violations": len(interface.contract_violations),
                    "api_version": interface.api_version,
                    "deprecation_warnings": len(interface.deprecation_warnings)
                }
                for module_id, interface in self.module_interfaces.items()
            },
            "interface_violations": [
                {
                    "type": violation.violation_type,
                    "severity": violation.severity,
                    "module": violation.module_id,
                    "description": violation.description,
                    "recommendation": violation.recommendation,
                    "impact": violation.impact_assessment
                }
                for violation in self.interface_violations
            ],
            "interface_contracts": {
                contract_id: {
                    "signature": contract.actual_signature,
                    "is_compatible": contract.is_compatible,
                    "deprecation_status": contract.deprecation_status,
                    "version": contract.version_introduced
                }
                for contract_id, contract in self.interface_contracts.items()
            },
            "interface_metrics": self.interface_metrics,
            "recommendations": self._generate_interface_recommendations()
        }
    
    def _generate_interface_recommendations(self) -> List[Dict[str, Any]]:
        """Generate interface improvement recommendations."""
        recommendations = []
        
        # Documentation improvement recommendations
        doc_coverage = self.interface_metrics.get('documentation_coverage', 0.0)
        if doc_coverage < 0.8:
            recommendations.append({
                "category": "documentation_improvement",
                "priority": "high",
                "description": f"Documentation coverage is {doc_coverage:.1%}, below 80% target",
                "actions": [
                    "Add docstrings to all public functions and classes",
                    "Include parameter descriptions and return value documentation",
                    "Add usage examples for complex interfaces"
                ]
            })
        
        # Interface stability recommendations
        avg_stability = self.interface_metrics.get('average_interface_stability', 0.0)
        if avg_stability < 0.7:
            recommendations.append({
                "category": "interface_stability",
                "priority": "medium",
                "description": f"Average interface stability is {avg_stability:.2f}, below 0.7 target",
                "actions": [
                    "Reduce complexity of high-complexity interfaces",
                    "Improve type hint coverage",
                    "Consider interface simplification for complex modules"
                ]
            })
        
        # Violation-based recommendations
        violation_count = len(self.interface_violations)
        if violation_count > 0:
            recommendations.append({
                "category": "violation_resolution",
                "priority": "high" if violation_count > 10 else "medium",
                "description": f"Found {violation_count} interface violations",
                "actions": [
                    "Resolve missing documentation violations",
                    "Add type hints where missing",
                    "Simplify complex interface signatures"
                ]
            })
        
        return recommendations
    
    def export_interface_documentation(self, output_file: str):
        """Export comprehensive interface documentation."""
        results = self._compile_interface_analysis_results()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Interface analysis results exported to {output_file}")

def main():
    """Run comprehensive interface analysis."""
    analyzer = InterfaceAnalyzer()
    
    logger.info("🚀 Starting Agent B Phase 2 Hours 36-40: Interface Analysis")
    
    # Perform comprehensive interface analysis
    results = analyzer.analyze_all_interfaces()
    
    # Export detailed results
    analyzer.export_interface_documentation("interface_analysis_results.json")
    
    # Print summary
    print(f"""
🎯 Interface Analysis Complete!

📊 Analysis Summary:
├── Modules Analyzed: {results['analysis_metadata']['modules_analyzed']}
├── Total Violations: {results['analysis_metadata']['total_violations']} 
├── Documentation Coverage: {results['interface_metrics'].get('documentation_coverage', 0.0):.1%}
├── Average Stability: {results['interface_metrics'].get('average_interface_stability', 0.0):.2f}
└── Recommendations Generated: {len(results['recommendations'])}

✅ Interface analysis results saved to interface_analysis_results.json
""")

if __name__ == "__main__":
    main()