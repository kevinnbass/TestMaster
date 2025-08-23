"""
Semantic Relationship Analysis Component
========================================

Analyzes semantic relationships between code elements.
Part of modularized semantic_analyzer system.
"""

import ast
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict

from .semantic_base import (
    SemanticRelationship, SemanticConfiguration
)


class SemanticRelationshipAnalyzer:
    """Analyzes semantic relationships between code elements"""
    
    def __init__(self, config: SemanticConfiguration):
        self.config = config
        self.relationships = []
    
    def analyze_semantic_relationships(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze semantic relationships between code elements"""
        relationships = {
            "direct_relationships": [],
            "indirect_relationships": [],
            "dependency_graph": {},
            "coupling_analysis": {}
        }
        
        # Build a map of all code elements
        element_map = {}
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            element_name = node.name
                            element_map[element_name] = {
                                "type": type(node).__name__,
                                "location": str(file_path),
                                "node": node
                            }
                            
            except Exception as e:
                print(f"Error mapping elements in {file_path}: {e}")
                
        # Analyze relationships
        for name, info in element_map.items():
            dependencies = self._extract_dependencies(info["node"])
            relationships["dependency_graph"][name] = dependencies
            
            for dep in dependencies:
                if dep in element_map:
                    relationship = SemanticRelationship(
                        from_element=name,
                        to_element=dep,
                        relationship_type="uses",
                        confidence=0.8,
                        context=f"Found in {info['location']}"
                    )
                    self.relationships.append(relationship)
                    relationships["direct_relationships"].append(relationship.to_dict())
                    
        # Analyze coupling
        relationships["coupling_analysis"] = self._analyze_coupling(element_map, relationships["dependency_graph"])
                    
        return relationships
    
    def analyze_naming_semantics(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze naming conventions and semantics"""
        naming_analysis = {
            "naming_conventions": defaultdict(list),
            "semantic_coherence": [],
            "naming_violations": [],
            "suggested_improvements": []
        }
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            name = node.name
                            convention = self._identify_naming_convention(name)
                            naming_analysis["naming_conventions"][convention].append(name)
                            
                            # Check semantic coherence
                            if isinstance(node, ast.FunctionDef):
                                coherence = self._check_function_name_coherence(node)
                                naming_analysis["semantic_coherence"].append({
                                    "name": name,
                                    "coherence": coherence,
                                    "location": str(file_path)
                                })
                                
                                if coherence < 0.7:
                                    naming_analysis["naming_violations"].append({
                                        "name": name,
                                        "issue": "Low semantic coherence",
                                        "suggestion": self._suggest_better_name(node),
                                        "location": str(file_path)
                                    })
                                    
            except Exception as e:
                print(f"Error analyzing naming in {file_path}: {e}")
                
        return naming_analysis
    
    def assess_semantic_quality(self, python_files: List[Path]) -> Dict[str, Any]:
        """Assess the semantic quality of code"""
        quality_assessment = {
            "clarity_score": 0.0,
            "consistency_score": 0.0,
            "expressiveness_score": 0.0,
            "overall_quality": 0.0,
            "improvement_suggestions": []
        }
        
        clarity_scores = []
        consistency_scores = []
        expressiveness_scores = []
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Assess clarity
                    clarity = self._assess_code_clarity(tree)
                    clarity_scores.append(clarity)
                    
                    # Assess consistency
                    consistency = self._assess_code_consistency(tree)
                    consistency_scores.append(consistency)
                    
                    # Assess expressiveness
                    expressiveness = self._assess_code_expressiveness(tree)
                    expressiveness_scores.append(expressiveness)
                    
            except Exception as e:
                print(f"Error assessing quality of {file_path}: {e}")
                
        # Calculate averages
        if clarity_scores:
            quality_assessment["clarity_score"] = sum(clarity_scores) / len(clarity_scores)
        if consistency_scores:
            quality_assessment["consistency_score"] = sum(consistency_scores) / len(consistency_scores)
        if expressiveness_scores:
            quality_assessment["expressiveness_score"] = sum(expressiveness_scores) / len(expressiveness_scores)
            
        quality_assessment["overall_quality"] = (
            quality_assessment["clarity_score"] +
            quality_assessment["consistency_score"] +
            quality_assessment["expressiveness_score"]
        ) / 3
        
        # Generate improvement suggestions
        if quality_assessment["clarity_score"] < 0.7:
            quality_assessment["improvement_suggestions"].append(
                "Improve code clarity through better naming and structure"
            )
        if quality_assessment["consistency_score"] < 0.7:
            quality_assessment["improvement_suggestions"].append(
                "Increase consistency in coding patterns and conventions"
            )
        if quality_assessment["expressiveness_score"] < 0.7:
            quality_assessment["improvement_suggestions"].append(
                "Make code more expressive and self-documenting"
            )
            
        return quality_assessment
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse a Python file into an AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ast.parse(f.read())
        except Exception:
            return None
    
    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extract dependencies from a node"""
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.append(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    dependencies.append(child.value.id)
        return list(set(dependencies))
    
    def _analyze_coupling(self, element_map: Dict, dependency_graph: Dict) -> Dict[str, Any]:
        """Analyze coupling between elements"""
        coupling_analysis = {
            "high_coupling": [],
            "low_coupling": [],
            "coupling_metrics": {},
            "suggestions": []
        }
        
        for element, dependencies in dependency_graph.items():
            coupling_score = len(dependencies)
            coupling_analysis["coupling_metrics"][element] = coupling_score
            
            if coupling_score > 10:
                coupling_analysis["high_coupling"].append({
                    "element": element,
                    "dependencies": len(dependencies),
                    "location": element_map.get(element, {}).get("location", "unknown")
                })
            elif coupling_score < 3:
                coupling_analysis["low_coupling"].append({
                    "element": element,
                    "dependencies": len(dependencies),
                    "location": element_map.get(element, {}).get("location", "unknown")
                })
        
        # Generate suggestions
        if coupling_analysis["high_coupling"]:
            coupling_analysis["suggestions"].append(
                "Consider refactoring highly coupled elements to reduce dependencies"
            )
        
        return coupling_analysis
    
    def _identify_naming_convention(self, name: str) -> str:
        """Identify the naming convention used"""
        import re
        
        for convention, pattern in self.config.naming_patterns.items():
            if re.match(pattern, name):
                return convention
        
        return "mixed_case"
    
    def _check_function_name_coherence(self, node: ast.FunctionDef) -> float:
        """Check if function name matches its behavior"""
        # Analyze function name vs content
        name_words = node.name.lower().replace('_', ' ').split()
        
        # Extract keywords from function body
        body_keywords = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                body_keywords.add(child.id.lower())
        
        # Calculate overlap
        name_keywords = set(name_words)
        overlap = len(name_keywords.intersection(body_keywords))
        total_name_words = len(name_keywords)
        
        return overlap / total_name_words if total_name_words > 0 else 0.5
    
    def _suggest_better_name(self, node: ast.FunctionDef) -> str:
        """Suggest a better name for a function"""
        # Extract action words from function body
        action_words = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                word = child.id.lower()
                if any(action in word for action in ["get", "set", "create", "update", "delete", "process"]):
                    action_words.append(word)
        
        if action_words:
            return f"{action_words[0]}_improved"
        return f"refactor_{node.name}"
    
    def _assess_code_clarity(self, tree: ast.AST) -> float:
        """Assess code clarity"""
        clarity_factors = []
        
        # Check for docstrings
        functions_with_docs = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if ast.get_docstring(node):
                    functions_with_docs += 1
        
        if total_functions > 0:
            doc_ratio = functions_with_docs / total_functions
            clarity_factors.append(doc_ratio)
        
        # Check for clear naming
        clear_names = 0
        total_names = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total_names += 1
                if len(node.name) > 3 and not node.name.startswith('_'):
                    clear_names += 1
        
        if total_names > 0:
            name_clarity = clear_names / total_names
            clarity_factors.append(name_clarity)
        
        return sum(clarity_factors) / len(clarity_factors) if clarity_factors else 0.5
    
    def _assess_code_consistency(self, tree: ast.AST) -> float:
        """Assess code consistency"""
        # Check naming convention consistency
        conventions = defaultdict(int)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                convention = self._identify_naming_convention(node.name)
                conventions[convention] += 1
        
        if conventions:
            max_convention_count = max(conventions.values())
            total_names = sum(conventions.values())
            consistency = max_convention_count / total_names
            return consistency
        
        return 0.5
    
    def _assess_code_expressiveness(self, tree: ast.AST) -> float:
        """Assess code expressiveness"""
        expressiveness_factors = []
        
        # Check for meaningful variable names
        meaningful_vars = 0
        total_vars = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                total_vars += 1
                if len(node.id) > 2 and not node.id.startswith('_'):
                    meaningful_vars += 1
        
        if total_vars > 0:
            var_expressiveness = meaningful_vars / total_vars
            expressiveness_factors.append(var_expressiveness)
        
        # Check for descriptive function names
        descriptive_funcs = 0
        total_funcs = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_funcs += 1
                if len(node.name) > 5 or '_' in node.name:
                    descriptive_funcs += 1
        
        if total_funcs > 0:
            func_expressiveness = descriptive_funcs / total_funcs
            expressiveness_factors.append(func_expressiveness)
        
        return sum(expressiveness_factors) / len(expressiveness_factors) if expressiveness_factors else 0.5
    
    def get_relationship_summary(self) -> Dict[str, Any]:
        """Get summary of relationship analysis"""
        return {
            "total_relationships": len(self.relationships),
            "relationship_types": list(set(r.relationship_type for r in self.relationships)),
            "average_confidence": sum(r.confidence for r in self.relationships) / 
                                 len(self.relationships) if self.relationships else 0.0
        }


# Export
__all__ = ['SemanticRelationshipAnalyzer']