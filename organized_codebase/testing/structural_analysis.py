"""
Structural Analysis Module
==========================

Implements comprehensive structural analysis:
- Design pattern detection (Singleton, Factory, Observer, etc.)
- Architectural pattern analysis
- Code structure assessment
- Module organization analysis
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class StructuralAnalyzer(BaseAnalyzer):
    """Analyzer for structural patterns in code."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive structural analysis."""
        print("[INFO] Analyzing Structural Patterns...")
        
        results = {
            "design_patterns": self._detect_design_patterns(),
            "architectural_patterns": self._analyze_architectural_patterns(),
            "module_organization": self._analyze_module_organization(),
            "code_structure": self._analyze_code_structure()
        }
        
        print(f"  [OK] Analyzed {len(results)} structural categories")
        return results
    
    def _detect_design_patterns(self) -> List[Dict[str, Any]]:
        """Detect common design patterns in the codebase."""
        patterns_found = []
        singleton_instances = []
        factory_instances = []
        observer_instances = []
        decorator_instances = []
        strategy_instances = []
        command_instances = []
        builder_instances = []
        adapter_instances = []
        
        for py_file in self._get_python_files():
            if not self._should_analyze_file(py_file):
                continue
                
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        
                        # Detect Singleton Pattern
                        if self._is_singleton_pattern(node, content):
                            singleton_instances.append({
                                "file": file_key,
                                "class": class_name,
                                "line": node.lineno,
                                "confidence": 0.9,
                                "evidence": self._get_singleton_evidence(node, content)
                            })
                        
                        # Detect Factory Pattern
                        if self._is_factory_pattern(node, content):
                            factory_instances.append({
                                "file": file_key,
                                "class": class_name,
                                "line": node.lineno,
                                "confidence": 0.8,
                                "evidence": self._get_factory_evidence(node, content)
                            })
                        
                        # Detect Observer Pattern
                        if self._is_observer_pattern(node, content):
                            observer_instances.append({
                                "file": file_key,
                                "class": class_name,
                                "line": node.lineno,
                                "confidence": 0.7,
                                "evidence": self._get_observer_evidence(node, content)
                            })
                        
                        # Detect Decorator Pattern
                        if self._is_decorator_pattern(node, content):
                            decorator_instances.append({
                                "file": file_key,
                                "class": class_name,
                                "line": node.lineno,
                                "confidence": 0.8,
                                "evidence": self._get_decorator_evidence(node, content)
                            })
                        
                        # Detect Strategy Pattern
                        if self._is_strategy_pattern(node, content):
                            strategy_instances.append({
                                "file": file_key,
                                "class": class_name,
                                "line": node.lineno,
                                "confidence": 0.7,
                                "evidence": self._get_strategy_evidence(node, content)
                            })
                        
                        # Detect Command Pattern
                        if self._is_command_pattern(node, content):
                            command_instances.append({
                                "file": file_key,
                                "class": class_name,
                                "line": node.lineno,
                                "confidence": 0.8,
                                "evidence": self._get_command_evidence(node, content)
                            })
                        
                        # Detect Builder Pattern
                        if self._is_builder_pattern(node, content):
                            builder_instances.append({
                                "file": file_key,
                                "class": class_name,
                                "line": node.lineno,
                                "confidence": 0.7,
                                "evidence": self._get_builder_evidence(node, content)
                            })
                        
                        # Detect Adapter Pattern
                        if self._is_adapter_pattern(node, content):
                            adapter_instances.append({
                                "file": file_key,
                                "class": class_name,
                                "line": node.lineno,
                                "confidence": 0.6,
                                "evidence": self._get_adapter_evidence(node, content)
                            })
            
            except Exception:
                continue
        
        # Compile results
        if singleton_instances:
            patterns_found.append({
                "pattern": "Singleton",
                "instances": len(singleton_instances),
                "locations": singleton_instances,
                "description": "Ensures a class has only one instance and provides global access",
                "benefits": "Controlled access to sole instance, reduced memory footprint",
                "drawbacks": "Difficult to unit test, violates single responsibility principle"
            })
        
        if factory_instances:
            patterns_found.append({
                "pattern": "Factory",
                "instances": len(factory_instances),
                "locations": factory_instances,
                "description": "Creates objects without specifying exact classes",
                "benefits": "Loose coupling, easy to extend with new types",
                "drawbacks": "Can become complex with many product types"
            })
        
        if observer_instances:
            patterns_found.append({
                "pattern": "Observer",
                "instances": len(observer_instances),
                "locations": observer_instances,
                "description": "Defines a subscription mechanism for notifications",
                "benefits": "Loose coupling between subject and observers",
                "drawbacks": "Can cause memory leaks if observers not properly removed"
            })
        
        if decorator_instances:
            patterns_found.append({
                "pattern": "Decorator",
                "instances": len(decorator_instances),
                "locations": decorator_instances,
                "description": "Adds behavior to objects without altering structure",
                "benefits": "More flexible than inheritance, follows open/closed principle",
                "drawbacks": "Can result in many small objects, complexity increases"
            })
        
        if strategy_instances:
            patterns_found.append({
                "pattern": "Strategy",
                "instances": len(strategy_instances),
                "locations": strategy_instances,
                "description": "Defines a family of algorithms and makes them interchangeable",
                "benefits": "Algorithms can vary independently from clients",
                "drawbacks": "Clients must be aware of different strategies"
            })
        
        if command_instances:
            patterns_found.append({
                "pattern": "Command",
                "instances": len(command_instances),
                "locations": command_instances,
                "description": "Encapsulates requests as objects for queuing and logging",
                "benefits": "Supports undo operations, logging, queuing",
                "drawbacks": "Increases complexity with many command classes"
            })
        
        if builder_instances:
            patterns_found.append({
                "pattern": "Builder",
                "instances": len(builder_instances),
                "locations": builder_instances,
                "description": "Constructs complex objects step by step",
                "benefits": "Flexible object construction, hides complex construction logic",
                "drawbacks": "Code complexity increases with many optional parameters"
            })
        
        if adapter_instances:
            patterns_found.append({
                "pattern": "Adapter",
                "instances": len(adapter_instances),
                "locations": adapter_instances,
                "description": "Allows incompatible interfaces to work together",
                "benefits": "Enables reuse of existing functionality",
                "drawbacks": "Increases overall complexity of code"
            })
        
        return patterns_found
    
    def _is_singleton_pattern(self, class_node: ast.ClassDef, content: str) -> bool:
        """Detect singleton pattern indicators."""
        indicators = 0
        
        # Look for instance variable
        if '_instance' in content.lower() or 'instance' in content.lower():
            indicators += 1
        
        # Look for __new__ method override
        has_new_method = any(isinstance(node, ast.FunctionDef) and node.name == '__new__' 
                           for node in class_node.body)
        if has_new_method:
            indicators += 2
        
        # Look for getInstance method
        has_get_instance = any(isinstance(node, ast.FunctionDef) and 'instance' in node.name.lower() 
                              for node in class_node.body)
        if has_get_instance:
            indicators += 1
        
        # Look for singleton in class or method names
        if 'singleton' in class_node.name.lower():
            indicators += 2
        
        return indicators >= 2
    
    def _get_singleton_evidence(self, class_node: ast.ClassDef, content: str) -> List[str]:
        """Get evidence for singleton pattern."""
        evidence = []
        
        if '_instance' in content:
            evidence.append("Uses _instance class variable")
        
        if any(isinstance(node, ast.FunctionDef) and node.name == '__new__' for node in class_node.body):
            evidence.append("Overrides __new__ method")
        
        if 'singleton' in class_node.name.lower():
            evidence.append("Contains 'singleton' in class name")
        
        return evidence
    
    def _is_factory_pattern(self, class_node: ast.ClassDef, content: str) -> bool:
        """Detect factory pattern indicators."""
        class_name = class_node.name.lower()
        
        # Factory naming patterns
        if 'factory' in class_name or 'creator' in class_name:
            return True
        
        # Look for creation methods
        creation_methods = ['create', 'make', 'build', 'get_instance', 'new_instance', 'produce']
        method_names = [node.name.lower() for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        creation_method_count = sum(1 for method in creation_methods if any(method in name for name in method_names))
        
        return creation_method_count >= 1
    
    def _get_factory_evidence(self, class_node: ast.ClassDef, content: str) -> List[str]:
        """Get evidence for factory pattern."""
        evidence = []
        
        if 'factory' in class_node.name.lower():
            evidence.append("Contains 'factory' in class name")
        
        creation_methods = ['create', 'make', 'build']
        method_names = [node.name.lower() for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        for method in creation_methods:
            if any(method in name for name in method_names):
                evidence.append(f"Has creation method containing '{method}'")
        
        return evidence
    
    def _is_observer_pattern(self, class_node: ast.ClassDef, content: str) -> bool:
        """Detect observer pattern indicators."""
        class_name = class_node.name.lower()
        
        # Observer naming patterns
        if any(keyword in class_name for keyword in ['observer', 'listener', 'subscriber', 'subject']):
            return True
        
        # Look for observer methods
        observer_methods = ['notify', 'update', 'on_changed', 'subscribe', 'unsubscribe', 'add_observer', 'remove_observer']
        method_names = [node.name.lower() for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        observer_method_count = sum(1 for method in observer_methods if method in method_names)
        
        return observer_method_count >= 2
    
    def _get_observer_evidence(self, class_node: ast.ClassDef, content: str) -> List[str]:
        """Get evidence for observer pattern."""
        evidence = []
        
        observer_keywords = ['observer', 'listener', 'subscriber', 'subject']
        for keyword in observer_keywords:
            if keyword in class_node.name.lower():
                evidence.append(f"Contains '{keyword}' in class name")
        
        observer_methods = ['notify', 'update', 'subscribe', 'unsubscribe']
        method_names = [node.name.lower() for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        for method in observer_methods:
            if method in method_names:
                evidence.append(f"Has '{method}' method")
        
        return evidence
    
    def _is_decorator_pattern(self, class_node: ast.ClassDef, content: str) -> bool:
        """Detect decorator pattern indicators."""
        class_name = class_node.name.lower()
        
        # Decorator naming patterns
        if 'decorator' in class_name or 'wrapper' in class_name:
            return True
        
        # Look for composition and delegation
        has_component_param = False
        has_delegation = False
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == '__init__':
                    # Look for component parameter (more than just 'self')
                    if len(node.args.args) > 1:
                        has_component_param = True
                
                # Look for delegation patterns in methods
                for n in ast.walk(node):
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute):
                        has_delegation = True
        
        return has_component_param and has_delegation
    
    def _get_decorator_evidence(self, class_node: ast.ClassDef, content: str) -> List[str]:
        """Get evidence for decorator pattern."""
        evidence = []
        
        if 'decorator' in class_node.name.lower():
            evidence.append("Contains 'decorator' in class name")
        
        if 'wrapper' in class_node.name.lower():
            evidence.append("Contains 'wrapper' in class name")
        
        # Check for composition in __init__
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                if len(node.args.args) > 1:
                    evidence.append("Takes component as constructor parameter")
        
        return evidence
    
    def _is_strategy_pattern(self, class_node: ast.ClassDef, content: str) -> bool:
        """Detect strategy pattern indicators."""
        class_name = class_node.name.lower()
        
        # Strategy naming patterns
        if 'strategy' in class_name or 'algorithm' in class_name or 'policy' in class_name:
            return True
        
        # Look for execute/perform methods
        execution_methods = ['execute', 'perform', 'process', 'apply', 'run']
        method_names = [node.name.lower() for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        return any(method in method_names for method in execution_methods)
    
    def _get_strategy_evidence(self, class_node: ast.ClassDef, content: str) -> List[str]:
        """Get evidence for strategy pattern."""
        evidence = []
        
        strategy_keywords = ['strategy', 'algorithm', 'policy']
        for keyword in strategy_keywords:
            if keyword in class_node.name.lower():
                evidence.append(f"Contains '{keyword}' in class name")
        
        execution_methods = ['execute', 'perform', 'process']
        method_names = [node.name.lower() for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        for method in execution_methods:
            if method in method_names:
                evidence.append(f"Has '{method}' method")
        
        return evidence
    
    def _is_command_pattern(self, class_node: ast.ClassDef, content: str) -> bool:
        """Detect command pattern indicators."""
        class_name = class_node.name.lower()
        
        # Command naming patterns
        if 'command' in class_name or 'action' in class_name:
            return True
        
        # Look for command methods
        command_methods = ['execute', 'undo', 'redo', 'do', 'call']
        method_names = [node.name.lower() for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        command_method_count = sum(1 for method in command_methods if method in method_names)
        
        return command_method_count >= 1
    
    def _get_command_evidence(self, class_node: ast.ClassDef, content: str) -> List[str]:
        """Get evidence for command pattern."""
        evidence = []
        
        if 'command' in class_node.name.lower():
            evidence.append("Contains 'command' in class name")
        
        command_methods = ['execute', 'undo', 'redo']
        method_names = [node.name.lower() for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        for method in command_methods:
            if method in method_names:
                evidence.append(f"Has '{method}' method")
        
        return evidence
    
    def _is_builder_pattern(self, class_node: ast.ClassDef, content: str) -> bool:
        """Detect builder pattern indicators."""
        class_name = class_node.name.lower()
        
        # Builder naming patterns
        if 'builder' in class_name:
            return True
        
        # Look for fluent interface (method chaining)
        method_names = [node.name for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        # Look for build method
        has_build_method = any('build' in name.lower() for name in method_names)
        
        # Look for setter-like methods that return self
        setter_like_methods = sum(1 for name in method_names 
                                 if any(prefix in name.lower() for prefix in ['set_', 'with_', 'add_']))
        
        return has_build_method and setter_like_methods >= 2
    
    def _get_builder_evidence(self, class_node: ast.ClassDef, content: str) -> List[str]:
        """Get evidence for builder pattern."""
        evidence = []
        
        if 'builder' in class_node.name.lower():
            evidence.append("Contains 'builder' in class name")
        
        method_names = [node.name for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        if any('build' in name.lower() for name in method_names):
            evidence.append("Has 'build' method")
        
        setter_methods = [name for name in method_names 
                         if any(prefix in name.lower() for prefix in ['set_', 'with_', 'add_'])]
        if len(setter_methods) >= 2:
            evidence.append(f"Has {len(setter_methods)} setter-like methods")
        
        return evidence
    
    def _is_adapter_pattern(self, class_node: ast.ClassDef, content: str) -> bool:
        """Detect adapter pattern indicators."""
        class_name = class_node.name.lower()
        
        # Adapter naming patterns
        if 'adapter' in class_name or 'wrapper' in class_name:
            return True
        
        # Look for composition (has another object as member)
        has_composition = False
        has_interface_methods = False
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == '__init__':
                    # Look for adaptee parameter
                    if len(node.args.args) > 1:
                        has_composition = True
                
                # Look for methods that delegate to internal object
                for n in ast.walk(node):
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute):
                        has_interface_methods = True
        
        return has_composition and has_interface_methods
    
    def _get_adapter_evidence(self, class_node: ast.ClassDef, content: str) -> List[str]:
        """Get evidence for adapter pattern."""
        evidence = []
        
        if 'adapter' in class_node.name.lower():
            evidence.append("Contains 'adapter' in class name")
        
        # Check for composition in __init__
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                if len(node.args.args) > 1:
                    evidence.append("Takes adaptee as constructor parameter")
        
        return evidence
    
    def _analyze_architectural_patterns(self) -> Dict[str, Any]:
        """Analyze architectural patterns in the codebase."""
        patterns = {
            'mvc_pattern': self._detect_mvc_pattern(),
            'layered_architecture': self._detect_layered_architecture(),
            'repository_pattern': self._detect_repository_pattern(),
            'dependency_injection': self._detect_dependency_injection()
        }
        
        return {
            'detected_patterns': patterns,
            'architecture_score': self._calculate_architecture_score(patterns),
            'recommendations': self._generate_architecture_recommendations(patterns)
        }
    
    def _detect_mvc_pattern(self) -> Dict[str, Any]:
        """Detect MVC (Model-View-Controller) pattern."""
        mvc_components = {
            'models': [],
            'views': [],
            'controllers': []
        }
        
        for py_file in self._get_python_files():
            file_name = py_file.name.lower()
            file_key = str(py_file.relative_to(self.base_path))
            
            # Look for MVC naming conventions
            if 'model' in file_name:
                mvc_components['models'].append(file_key)
            elif 'view' in file_name:
                mvc_components['views'].append(file_key)
            elif 'controller' in file_name:
                mvc_components['controllers'].append(file_key)
            
            # Look for MVC directories
            parts = py_file.parts
            if any('model' in part.lower() for part in parts):
                mvc_components['models'].append(file_key)
            elif any('view' in part.lower() for part in parts):
                mvc_components['views'].append(file_key)
            elif any('controller' in part.lower() for part in parts):
                mvc_components['controllers'].append(file_key)
        
        has_mvc = all(len(components) > 0 for components in mvc_components.values())
        
        return {
            'detected': has_mvc,
            'components': mvc_components,
            'completeness': sum(1 for comp in mvc_components.values() if comp) / 3,
            'confidence': 0.8 if has_mvc else 0.3
        }
    
    def _detect_layered_architecture(self) -> Dict[str, Any]:
        """Detect layered architecture pattern."""
        layers = {
            'presentation': [],
            'business': [],
            'data': [],
            'service': []
        }
        
        layer_keywords = {
            'presentation': ['ui', 'view', 'template', 'frontend', 'web'],
            'business': ['business', 'logic', 'domain', 'core', 'service'],
            'data': ['data', 'model', 'entity', 'repository', 'dao', 'database'],
            'service': ['service', 'api', 'endpoint', 'handler']
        }
        
        for py_file in self._get_python_files():
            file_path = str(py_file).lower()
            file_key = str(py_file.relative_to(self.base_path))
            
            for layer_name, keywords in layer_keywords.items():
                if any(keyword in file_path for keyword in keywords):
                    layers[layer_name].append(file_key)
                    break
        
        layer_count = sum(1 for layer_files in layers.values() if layer_files)
        
        return {
            'detected': layer_count >= 2,
            'layers': layers,
            'layer_count': layer_count,
            'separation_score': layer_count / 4,
            'confidence': 0.7 if layer_count >= 3 else 0.4
        }
    
    def _detect_repository_pattern(self) -> Dict[str, Any]:
        """Detect repository pattern."""
        repositories = []
        
        for py_file in self._get_python_files():
            file_name = py_file.name.lower()
            file_key = str(py_file.relative_to(self.base_path))
            
            if 'repository' in file_name or 'repo' in file_name:
                repositories.append({
                    'file': file_key,
                    'type': 'naming_convention'
                })
            
            # Look for repository-like classes
            try:
                tree = self._get_ast(py_file)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name.lower()
                        if 'repository' in class_name or 'repo' in class_name:
                            repositories.append({
                                'file': file_key,
                                'class': node.name,
                                'line': node.lineno,
                                'type': 'class_name'
                            })
            except:
                continue
        
        return {
            'detected': len(repositories) > 0,
            'repositories': repositories,
            'count': len(repositories),
            'confidence': 0.8 if repositories else 0.1
        }
    
    def _detect_dependency_injection(self) -> Dict[str, Any]:
        """Detect dependency injection pattern."""
        di_indicators = []
        
        # Look for constructor injection
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for method in node.body:
                            if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                                # Look for dependency injection indicators
                                param_count = len(method.args.args) - 1  # Exclude 'self'
                                if param_count >= 2:  # Likely injecting dependencies
                                    di_indicators.append({
                                        'file': file_key,
                                        'class': node.name,
                                        'line': node.lineno,
                                        'parameter_count': param_count,
                                        'type': 'constructor_injection'
                                    })
            except:
                continue
        
        return {
            'detected': len(di_indicators) > 0,
            'indicators': di_indicators,
            'count': len(di_indicators),
            'confidence': 0.6 if di_indicators else 0.2
        }
    
    def _calculate_architecture_score(self, patterns: Dict[str, Any]) -> float:
        """Calculate overall architecture score."""
        scores = []
        
        for pattern_data in patterns.values():
            if pattern_data.get('detected', False):
                confidence = pattern_data.get('confidence', 0.5)
                scores.append(confidence)
        
        return sum(scores) / max(len(patterns), 1)
    
    def _generate_architecture_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate architecture recommendations."""
        recommendations = []
        
        if not patterns['mvc_pattern']['detected']:
            recommendations.append("Consider implementing MVC pattern for better separation of concerns")
        
        if patterns['layered_architecture']['layer_count'] < 3:
            recommendations.append("Consider implementing a layered architecture for better modularity")
        
        if not patterns['repository_pattern']['detected']:
            recommendations.append("Consider using Repository pattern for data access abstraction")
        
        if not patterns['dependency_injection']['detected']:
            recommendations.append("Consider implementing dependency injection for better testability")
        
        return recommendations
    
    def _analyze_module_organization(self) -> Dict[str, Any]:
        """Analyze module organization and structure."""
        module_stats = {
            'total_modules': 0,
            'average_module_size': 0,
            'modules_by_size': defaultdict(int),
            'package_structure': {},
            'import_relationships': defaultdict(set)
        }
        
        module_sizes = []
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                size = len(content.split('\n'))
                module_sizes.append(size)
                
                # Categorize by size
                if size < 50:
                    module_stats['modules_by_size']['small'] += 1
                elif size < 200:
                    module_stats['modules_by_size']['medium'] += 1
                elif size < 500:
                    module_stats['modules_by_size']['large'] += 1
                else:
                    module_stats['modules_by_size']['very_large'] += 1
                
                # Analyze package structure
                parts = py_file.parts[:-1]  # Exclude filename
                package_path = '/'.join(parts)
                if package_path not in module_stats['package_structure']:
                    module_stats['package_structure'][package_path] = 0
                module_stats['package_structure'][package_path] += 1
                
                # Analyze imports
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_stats['import_relationships'][file_key].add(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        module_stats['import_relationships'][file_key].add(node.module)
                
            except:
                continue
        
        module_stats['total_modules'] = len(module_sizes)
        module_stats['average_module_size'] = sum(module_sizes) / max(len(module_sizes), 1)
        
        return module_stats
    
    def _analyze_code_structure(self) -> Dict[str, Any]:
        """Analyze overall code structure."""
        structure_metrics = {
            'class_count': 0,
            'function_count': 0,
            'average_class_methods': 0,
            'inheritance_depth': [],
            'complexity_distribution': defaultdict(int)
        }
        
        class_method_counts = []
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        structure_metrics['class_count'] += 1
                        
                        # Count methods in class
                        method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                        class_method_counts.append(method_count)
                        
                        # Inheritance depth (simplified)
                        inheritance_depth = len(node.bases)
                        structure_metrics['inheritance_depth'].append(inheritance_depth)
                    
                    elif isinstance(node, ast.FunctionDef):
                        structure_metrics['function_count'] += 1
                        
                        # Function complexity
                        complexity = self._calculate_function_complexity(node)
                        if complexity <= 5:
                            structure_metrics['complexity_distribution']['simple'] += 1
                        elif complexity <= 10:
                            structure_metrics['complexity_distribution']['moderate'] += 1
                        elif complexity <= 20:
                            structure_metrics['complexity_distribution']['complex'] += 1
                        else:
                            structure_metrics['complexity_distribution']['very_complex'] += 1
                
            except:
                continue
        
        if class_method_counts:
            structure_metrics['average_class_methods'] = sum(class_method_counts) / len(class_method_counts)
        
        return structure_metrics