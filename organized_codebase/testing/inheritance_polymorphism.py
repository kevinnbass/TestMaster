"""
Inheritance and Polymorphism Analyzer
====================================

Implements inheritance and polymorphism metrics:
- Depth of Inheritance Tree (DIT)
- Number of Children (NOC)
- Polymorphism analysis
- Interface usage analysis
"""

import ast
import statistics
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict

from .base_analyzer import BaseAnalyzer


class InheritancePolymorphismAnalyzer(BaseAnalyzer):
    """Analyzer for inheritance and polymorphism metrics."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive inheritance and polymorphism analysis."""
        return {
            "inheritance_metrics": self._calculate_inheritance_metrics(),
            "polymorphism_metrics": self._calculate_polymorphism_metrics()
        }
    
    def _calculate_inheritance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive inheritance metrics (DIT, NOC)."""
        inheritance_data = {}
        class_hierarchy = defaultdict(list)  # parent -> [children]
        class_parents = {}  # child -> parent
        all_classes = {}  # class_name -> file_info
        
        # Build class hierarchy
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_key = f"{file_key}::{node.name}"
                        all_classes[class_key] = {
                            "file": file_key,
                            "name": node.name,
                            "line": node.lineno,
                            "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                        }
                        
                        # Track inheritance relationships
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                parent_name = base.id
                                # Try to resolve parent class
                                parent_key = self._resolve_class_name(parent_name, file_key, all_classes)
                                if parent_key:
                                    class_hierarchy[parent_key].append(class_key)
                                    class_parents[class_key] = parent_key
                                    
            except Exception:
                continue
        
        # Calculate metrics for each class
        dit_scores = []
        noc_scores = []
        
        for class_key, class_info in all_classes.items():
            # Calculate Depth of Inheritance Tree (DIT)
            dit = self._calculate_dit(class_key, class_parents)
            
            # Calculate Number of Children (NOC)
            noc = len(class_hierarchy.get(class_key, []))
            
            inheritance_data[class_key] = {
                **class_info,
                "dit": dit,
                "noc": noc,
                "children": class_hierarchy.get(class_key, []),
                "parent": class_parents.get(class_key)
            }
            
            dit_scores.append(dit)
            noc_scores.append(noc)
        
        # Calculate summary statistics
        if dit_scores and noc_scores:
            return {
                "per_class": inheritance_data,
                "class_count": len(all_classes),
                "inheritance_relationships": len(class_parents),
                "summary": {
                    "average_dit": statistics.mean(dit_scores),
                    "max_dit": max(dit_scores),
                    "average_noc": statistics.mean(noc_scores),
                    "max_noc": max(noc_scores),
                    "deep_inheritance_classes": len([dit for dit in dit_scores if dit > 3]),
                    "highly_inherited_classes": len([noc for noc in noc_scores if noc > 5]),
                    "root_classes": len([key for key in all_classes if key not in class_parents]),
                    "leaf_classes": len([key for key in all_classes if not class_hierarchy.get(key)])
                }
            }
        else:
            return {"per_class": {}, "class_count": 0, "summary": {}}
    
    def _resolve_class_name(self, class_name: str, current_file: str, all_classes: Dict[str, Any]) -> str:
        """Try to resolve a class name to its full key."""
        # First try same file
        same_file_key = f"{current_file}::{class_name}"
        if same_file_key in all_classes:
            return same_file_key
        
        # Then try other files (simplified resolution)
        for key in all_classes:
            if key.endswith(f"::{class_name}"):
                return key
        
        return None
    
    def _calculate_dit(self, class_key: str, class_parents: Dict[str, str]) -> int:
        """Calculate Depth of Inheritance Tree for a class."""
        depth = 0
        current = class_key
        visited = set()
        
        while current in class_parents and current not in visited:
            visited.add(current)
            current = class_parents[current]
            depth += 1
            if depth > 10:  # Prevent infinite loops
                break
        
        return depth
    
    def _calculate_polymorphism_metrics(self) -> Dict[str, Any]:
        """Calculate polymorphism metrics."""
        polymorphism_data = {}
        method_overrides = defaultdict(list)
        abstract_methods = defaultdict(list)
        interface_implementations = defaultdict(list)
        
        # Find method overrides and polymorphic behavior
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_key = f"{file_key}::{node.name}"
                        
                        # Find methods in this class
                        class_methods = {}
                        abstract_method_count = 0
                        
                        for method_node in node.body:
                            if isinstance(method_node, ast.FunctionDef):
                                method_name = method_node.name
                                class_methods[method_name] = {
                                    "line": method_node.lineno,
                                    "args": len(method_node.args.args),
                                    "is_abstract": self._is_abstract_method(method_node)
                                }
                                
                                if class_methods[method_name]["is_abstract"]:
                                    abstract_method_count += 1
                                    abstract_methods[class_key].append(method_name)
                        
                        # Detect method overrides (simplified heuristic)
                        common_methods = ["__init__", "__str__", "__repr__", "__eq__", "__hash__"]
                        overridden_methods = [method for method in class_methods if method in common_methods]
                        
                        # Check for interface-like behavior
                        is_interface_like = (
                            abstract_method_count > 0 and
                            abstract_method_count / max(len(class_methods), 1) > 0.5
                        )
                        
                        polymorphism_data[class_key] = {
                            "file": file_key,
                            "class_name": node.name,
                            "total_methods": len(class_methods),
                            "abstract_methods": abstract_method_count,
                            "overridden_methods": len(overridden_methods),
                            "override_details": overridden_methods,
                            "is_interface_like": is_interface_like,
                            "polymorphism_ratio": len(overridden_methods) / max(len(class_methods), 1)
                        }
                        
                        if overridden_methods:
                            method_overrides[class_key] = overridden_methods
                            
                        if is_interface_like:
                            interface_implementations[class_key] = abstract_methods[class_key]
                
            except Exception:
                continue
        
        # Calculate summary statistics
        if polymorphism_data:
            polymorphism_ratios = [data["polymorphism_ratio"] for data in polymorphism_data.values()]
            abstract_counts = [data["abstract_methods"] for data in polymorphism_data.values()]
            
            return {
                "per_class": polymorphism_data,
                "method_overrides": dict(method_overrides),
                "abstract_methods": dict(abstract_methods),
                "interface_implementations": dict(interface_implementations),
                "summary": {
                    "classes_analyzed": len(polymorphism_data),
                    "classes_with_overrides": len(method_overrides),
                    "classes_with_abstracts": len(abstract_methods),
                    "interface_like_classes": len(interface_implementations),
                    "average_polymorphism_ratio": statistics.mean(polymorphism_ratios) if polymorphism_ratios else 0,
                    "max_abstract_methods": max(abstract_counts) if abstract_counts else 0,
                    "total_method_overrides": sum(len(overrides) for overrides in method_overrides.values()),
                    "polymorphic_classes": len([ratio for ratio in polymorphism_ratios if ratio > 0.3])
                }
            }
        else:
            return {"per_class": {}, "summary": {"classes_analyzed": 0}}
    
    def _is_abstract_method(self, method_node: ast.FunctionDef) -> bool:
        """Check if a method is abstract (simplified detection)."""
        # Look for @abstractmethod decorator
        for decorator in method_node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                return True
            elif isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
                return True
        
        # Look for NotImplementedError or pass-only body
        if len(method_node.body) == 1:
            first_stmt = method_node.body[0]
            if isinstance(first_stmt, ast.Pass):
                return True
            elif isinstance(first_stmt, ast.Raise):
                if isinstance(first_stmt.exc, ast.Call):
                    if isinstance(first_stmt.exc.func, ast.Name):
                        if first_stmt.exc.func.id == "NotImplementedError":
                            return True
        
        return False