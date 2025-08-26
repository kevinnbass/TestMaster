"""
Semantic Pattern Detection Component
====================================

Detects conceptual patterns, design patterns, and anti-patterns in code.
Part of modularized semantic_analyzer system.
"""

import ast
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict

from .semantic_base import (
    ConceptualPattern, SemanticConfiguration
)


class SemanticPatternDetector:
    """Detects semantic patterns in code"""
    
    def __init__(self, config: SemanticConfiguration):
        self.config = config
        self.conceptual_patterns = []
    
    def identify_conceptual_patterns(self, python_files: List[Path]) -> Dict[str, Any]:
        """Identify conceptual patterns in code"""
        patterns = {
            "design_patterns": [],
            "architectural_patterns": [],
            "idioms": [],
            "anti_patterns": []
        }
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Check for design patterns
                    for pattern_name, pattern_info in self.config.semantic_patterns.items():
                        if self._check_pattern_presence(tree, pattern_info["indicators"]):
                            patterns["design_patterns"].append({
                                "pattern": pattern_name,
                                "location": str(file_path),
                                "role": pattern_info["role"],
                                "confidence": self._calculate_pattern_confidence(tree, pattern_info)
                            })
                            
                    # Check for anti-patterns
                    anti_patterns = self._detect_anti_patterns(tree)
                    if anti_patterns:
                        patterns["anti_patterns"].extend([{
                            "pattern": ap,
                            "location": str(file_path)
                        } for ap in anti_patterns])
                        
            except Exception as e:
                print(f"Error identifying patterns in {file_path}: {e}")
                
        return patterns
    
    def identify_behavioral_patterns(self, python_files: List[Path]) -> Dict[str, Any]:
        """Identify behavioral patterns in code"""
        behavioral_patterns = {
            "state_machines": [],
            "event_driven": [],
            "pipeline_patterns": [],
            "callback_patterns": []
        }
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Detect state machine patterns
                    if self._has_state_machine_pattern(tree):
                        behavioral_patterns["state_machines"].append(str(file_path))
                        
                    # Detect event-driven patterns
                    if self._has_event_pattern(tree):
                        behavioral_patterns["event_driven"].append(str(file_path))
                        
                    # Detect pipeline patterns
                    if self._has_pipeline_pattern(tree):
                        behavioral_patterns["pipeline_patterns"].append(str(file_path))
                        
                    # Detect callback patterns
                    if self._has_callback_pattern(tree):
                        behavioral_patterns["callback_patterns"].append(str(file_path))
                        
            except Exception as e:
                print(f"Error identifying behavioral patterns in {file_path}: {e}")
                
        return behavioral_patterns
    
    def extract_domain_concepts(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract domain-specific concepts from code"""
        domain_concepts = {
            "entities": [],
            "value_objects": [],
            "services": [],
            "repositories": [],
            "domain_events": []
        }
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Classify domain element
                            domain_type = self._classify_domain_element(node)
                            if domain_type:
                                domain_concepts[domain_type].append({
                                    "name": node.name,
                                    "location": str(file_path),
                                    "properties": self._extract_class_properties(node)
                                })
                                
            except Exception as e:
                print(f"Error extracting domain concepts from {file_path}: {e}")
                
        return domain_concepts
    
    def perform_semantic_clustering(self, python_files: List[Path]) -> Dict[str, Any]:
        """Cluster code elements based on semantic similarity"""
        clusters = {
            "semantic_clusters": [],
            "cluster_coherence": [],
            "outliers": []
        }
        
        # Collect all code elements with their semantic features
        elements = []
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            features = self._extract_semantic_features(node)
                            elements.append({
                                "name": node.name,
                                "type": type(node).__name__,
                                "features": features,
                                "location": str(file_path)
                            })
                            
            except Exception as e:
                print(f"Error collecting elements from {file_path}: {e}")
                
        # Simple clustering based on feature similarity
        if elements:
            clusters["semantic_clusters"] = self._simple_cluster(elements)
            
        return clusters
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse a Python file into an AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ast.parse(f.read())
        except Exception:
            return None
    
    def _check_pattern_presence(self, tree: ast.AST, indicators: List[str]) -> bool:
        """Check if pattern indicators are present in AST"""
        code_str = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        return any(indicator in code_str.lower() for indicator in indicators)
    
    def _calculate_pattern_confidence(self, tree: ast.AST, pattern_info: Dict) -> float:
        """Calculate confidence for a pattern match"""
        indicators_found = 0
        for indicator in pattern_info["indicators"]:
            if self._check_pattern_presence(tree, [indicator]):
                indicators_found += 1
        return indicators_found / len(pattern_info["indicators"]) if pattern_info["indicators"] else 0.0
    
    def _detect_anti_patterns(self, tree: ast.AST) -> List[str]:
        """Detect anti-patterns in code"""
        anti_patterns = []
        
        # Check for God Class
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        for cls in classes:
            method_count = sum(1 for n in cls.body if isinstance(n, ast.FunctionDef))
            if method_count > self.config.anti_pattern_thresholds["god_class_methods"]:
                anti_patterns.append(f"GodClass:{cls.name}")
                
        # Check for Long Method
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        for func in functions:
            if len(func.body) > self.config.anti_pattern_thresholds["long_method_lines"]:
                anti_patterns.append(f"LongMethod:{func.name}")
                
        # Check for Deep Nesting
        for func in functions:
            max_depth = self._calculate_nesting_depth(func)
            if max_depth > self.config.anti_pattern_thresholds["deep_nesting"]:
                anti_patterns.append(f"DeepNesting:{func.name}")
                
        return anti_patterns
    
    def _has_state_machine_pattern(self, tree: ast.AST) -> bool:
        """Check if code has state machine pattern"""
        code_str = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        return any(pattern in code_str.lower() for pattern in ["state", "transition", "fsm"])
    
    def _has_event_pattern(self, tree: ast.AST) -> bool:
        """Check if code has event-driven pattern"""
        code_str = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        return any(pattern in code_str.lower() for pattern in ["event", "listener", "emit", "subscribe"])
    
    def _has_pipeline_pattern(self, tree: ast.AST) -> bool:
        """Check if code has pipeline pattern"""
        code_str = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        return any(pattern in code_str.lower() for pattern in ["pipeline", "chain", "flow", "stream"])
    
    def _has_callback_pattern(self, tree: ast.AST) -> bool:
        """Check if code has callback pattern"""
        code_str = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        return any(pattern in code_str.lower() for pattern in ["callback", "hook", "handler", "delegate"])
    
    def _classify_domain_element(self, node: ast.ClassDef) -> Optional[str]:
        """Classify a class as a domain element"""
        name_lower = node.name.lower()
        
        if "entity" in name_lower:
            return "entities"
        elif "value" in name_lower:
            return "value_objects"
        elif "service" in name_lower:
            return "services"
        elif "repository" in name_lower:
            return "repositories"
        elif "event" in name_lower:
            return "domain_events"
        
        return None
    
    def _extract_class_properties(self, node: ast.ClassDef) -> List[str]:
        """Extract properties/attributes from a class"""
        properties = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        properties.append(target.id)
        return properties
    
    def _extract_semantic_features(self, node: ast.AST) -> Dict[str, Any]:
        """Extract semantic features from a node"""
        return {
            "type": type(node).__name__,
            "size": len(ast.walk(node)),
            "complexity": self._calculate_complexity(node),
            "nesting_depth": self._calculate_nesting_depth(node)
        }
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        return get_depth(node)
    
    def _simple_cluster(self, elements: List[Dict]) -> List[List[Dict]]:
        """Simple clustering of elements"""
        # Group by type and similar complexity
        clusters = defaultdict(list)
        for element in elements:
            features = element["features"]
            cluster_key = f"{element['type']}_{features.get('complexity', 0)//5}"
            clusters[cluster_key].append(element)
        return list(clusters.values())
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns"""
        return {
            "total_patterns": len(self.conceptual_patterns),
            "pattern_types": list(set(p.pattern_name for p in self.conceptual_patterns)),
            "average_quality": sum(p.implementation_quality for p in self.conceptual_patterns) / 
                            len(self.conceptual_patterns) if self.conceptual_patterns else 0.0
        }


# Export
__all__ = ['SemanticPatternDetector']