"""
Coupling and Cohesion Analyzer
==============================

Implements comprehensive coupling and cohesion metrics:
- Efferent/Afferent Coupling (fan-out/fan-in)
- Instability metrics
- LCOM (Lack of Cohesion of Methods)
- Class cohesion analysis
"""

import ast
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .base_analyzer import BaseAnalyzer


class CouplingCohesionAnalyzer(BaseAnalyzer):
    """Analyzer for coupling and cohesion metrics."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive coupling and cohesion analysis."""
        return {
            "coupling_metrics": self._calculate_coupling_metrics(),
            "cohesion_metrics": self._calculate_cohesion_metrics()
        }
    
    def _calculate_coupling_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive coupling metrics (fan-in, fan-out, CBO)."""
        coupling_data = {}
        import_graph = defaultdict(set)
        class_dependencies = defaultdict(set)
        
        # Build import and dependency relationships
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Track imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            import_graph[file_key].add(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        import_graph[file_key].add(node.module)
                    
                    # Track class dependencies (inheritance, composition)
                    elif isinstance(node, ast.ClassDef):
                        class_name = f"{file_key}::{node.name}"
                        # Inheritance dependencies
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                class_dependencies[class_name].add(base.id)
                        
                        # Method calls and attribute access (composition)
                        for method in node.body:
                            if isinstance(method, ast.FunctionDef):
                                for n in ast.walk(method):
                                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                                        if isinstance(n.func.value, ast.Name):
                                            class_dependencies[class_name].add(n.func.value.id)
                        
            except Exception:
                continue
        
        # Calculate coupling metrics for each file
        for file_key in import_graph.keys() or []:
            efferent_coupling = len(import_graph[file_key])  # Fan-out
            afferent_coupling = sum(1 for other_file, imports in import_graph.items() 
                                  if other_file != file_key and 
                                  any(file_key.replace('.py', '').replace('/', '.') in imp or 
                                     imp.replace('/', '.') in file_key for imp in imports))  # Fan-in
            
            # Calculate instability (I = Ce / (Ca + Ce))
            total_coupling = efferent_coupling + afferent_coupling
            instability = efferent_coupling / max(total_coupling, 1)
            
            coupling_data[file_key] = {
                "efferent_coupling": efferent_coupling,  # Dependencies out
                "afferent_coupling": afferent_coupling,  # Dependencies in
                "instability": instability,  # 0 = stable, 1 = unstable
                "total_coupling": total_coupling,
                "imported_modules": list(import_graph[file_key])
            }
        
        # Calculate summary statistics
        if coupling_data:
            efferent_values = [data["efferent_coupling"] for data in coupling_data.values()]
            afferent_values = [data["afferent_coupling"] for data in coupling_data.values()]
            instability_values = [data["instability"] for data in coupling_data.values()]
            
            return {
                "per_file": coupling_data,
                "summary": {
                    "average_efferent_coupling": statistics.mean(efferent_values),
                    "average_afferent_coupling": statistics.mean(afferent_values),
                    "average_instability": statistics.mean(instability_values),
                    "max_efferent_coupling": max(efferent_values),
                    "max_afferent_coupling": max(afferent_values),
                    "highly_coupled_files": len([f for f, data in coupling_data.items() 
                                               if data["total_coupling"] > 10]),
                    "unstable_files": len([f for f, data in coupling_data.items() 
                                         if data["instability"] > 0.7]),
                    "stable_files": len([f for f, data in coupling_data.items() 
                                       if data["instability"] < 0.3])
                }
            }
        else:
            return {"per_file": {}, "summary": {"average_efferent_coupling": 0}}
    
    def _calculate_cohesion_metrics(self) -> Dict[str, Any]:
        """Calculate cohesion metrics using LCOM (Lack of Cohesion of Methods)."""
        cohesion_data = {}
        all_lcom_scores = []
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                
                file_key = str(py_file.relative_to(self.base_path))
                class_cohesion = {}
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        lcom_score = self._calculate_lcom(node)
                        class_cohesion[node.name] = lcom_score
                        if lcom_score is not None:
                            all_lcom_scores.append(lcom_score)
                
                if class_cohesion:
                    cohesion_data[file_key] = class_cohesion
                    
            except Exception:
                continue
        
        # Calculate summary statistics
        if all_lcom_scores:
            return {
                "per_class": cohesion_data,
                "summary": {
                    "average_lcom": statistics.mean(all_lcom_scores),
                    "median_lcom": statistics.median(all_lcom_scores),
                    "max_lcom": max(all_lcom_scores),
                    "min_lcom": min(all_lcom_scores),
                    "classes_analyzed": len(all_lcom_scores),
                    "low_cohesion_classes": len([score for score in all_lcom_scores if score > 1]),
                    "high_cohesion_classes": len([score for score in all_lcom_scores if score == 0]),
                    "cohesion_distribution": self._calculate_distribution(all_lcom_scores)
                }
            }
        else:
            return {"per_class": {}, "summary": {"average_lcom": 0, "classes_analyzed": 0}}
    
    def _calculate_lcom(self, class_node: ast.ClassDef) -> Optional[float]:
        """Calculate LCOM (Lack of Cohesion of Methods) for a class."""
        methods = []
        instance_variables = set()
        
        # Extract methods and identify instance variables
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('__'):
                method_variables = set()
                
                # Find instance variables used in this method
                for n in ast.walk(node):
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'self':
                        instance_variables.add(n.attr)
                        method_variables.add(n.attr)
                    elif isinstance(n, ast.Assign):
                        for target in n.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                instance_variables.add(target.attr)
                                method_variables.add(target.attr)
                
                methods.append({
                    'name': node.name,
                    'variables': method_variables
                })
        
        if len(methods) < 2:
            return 0  # Perfect cohesion for classes with 0-1 methods
        
        # Calculate LCOM using Henderson-Sellers method
        if not instance_variables:
            return len(methods)  # No shared variables = maximum lack of cohesion
        
        methods_per_variable = {}
        for var in instance_variables:
            methods_per_variable[var] = sum(1 for method in methods if var in method['variables'])
        
        if not methods_per_variable:
            return len(methods)
        
        sum_mv = sum(methods_per_variable.values())
        M = len(methods)
        V = len(instance_variables)
        
        if V == 0:
            return M
        
        lcom = (M - sum_mv/V) / (M - 1) if M > 1 else 0
        return max(0, lcom)  # LCOM should not be negative