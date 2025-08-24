"""
Cognitive Load Analysis Module
Analyzes code cognitive complexity beyond traditional metrics
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import math
import json

from testmaster.analysis.base_analyzer import BaseAnalyzer


@dataclass
class CognitiveMetrics:
    """Cognitive load metrics for a code element"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    halstead_metrics: Dict[str, float]
    nesting_depth: int
    parameter_count: int
    variable_scope_count: int
    control_flow_breaks: int
    cognitive_load_score: float
    readability_score: float
    maintainability_index: float


class CognitiveLoadAnalyzer(BaseAnalyzer):
    """
    Analyzes cognitive load and complexity in code
    Goes beyond cyclomatic complexity to measure actual mental effort
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.cognitive_metrics = {}
        self.complexity_thresholds = {
            "low": 10,
            "medium": 20,
            "high": 40,
            "very_high": 60
        }
        self.readability_factors = {
            "naming_quality": 0.2,
            "structure_clarity": 0.3,
            "documentation_presence": 0.15,
            "consistency": 0.15,
            "abstraction_level": 0.2
        }
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive cognitive load analysis
        """
        results = {
            "cognitive_complexity": self._analyze_cognitive_complexity(),
            "halstead_metrics": self._calculate_halstead_metrics(),
            "readability_analysis": self._analyze_readability(),
            "maintainability_index": self._calculate_maintainability_index(),
            "nesting_analysis": self._analyze_nesting_depth(),
            "naming_quality": self._analyze_naming_quality(),
            "abstraction_analysis": self._analyze_abstraction_levels(),
            "pattern_consistency": self._analyze_pattern_consistency(),
            "mental_model_complexity": self._analyze_mental_models(),
            "code_shape_analysis": self._analyze_code_shape(),
            "refactoring_suggestions": self._generate_refactoring_suggestions(),
            "cognitive_hotspots": self._identify_cognitive_hotspots(),
            "comprehension_barriers": self._identify_comprehension_barriers(),
            "learning_curve_estimation": self._estimate_learning_curve(),
            "summary": self._generate_summary()
        }
        
        return results
    
    def _analyze_cognitive_complexity(self) -> Dict[str, Any]:
        """
        Analyze cognitive complexity using Cognitive Complexity metric
        More accurate than cyclomatic complexity for measuring understandability
        """
        cognitive_analysis = {
            "function_complexities": [],
            "class_complexities": [],
            "module_complexities": [],
            "complexity_distribution": defaultdict(int),
            "high_complexity_areas": [],
            "complexity_trends": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    module_complexity = self._calculate_module_cognitive_complexity(tree)
                    cognitive_analysis["module_complexities"].append({
                        "file": str(file_path),
                        "complexity": module_complexity,
                        "category": self._categorize_complexity(module_complexity)
                    })
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_complexity = self._calculate_cognitive_complexity(node)
                            cognitive_analysis["function_complexities"].append({
                                "function": node.name,
                                "file": str(file_path),
                                "complexity": func_complexity,
                                "cyclomatic": self._calculate_cyclomatic_complexity(node),
                                "cognitive": func_complexity,
                                "difference": func_complexity - self._calculate_cyclomatic_complexity(node),
                                "category": self._categorize_complexity(func_complexity)
                            })
                            
                            if func_complexity > self.complexity_thresholds["high"]:
                                cognitive_analysis["high_complexity_areas"].append({
                                    "type": "function",
                                    "name": node.name,
                                    "file": str(file_path),
                                    "complexity": func_complexity,
                                    "reasons": self._analyze_complexity_reasons(node)
                                })
                                
                        elif isinstance(node, ast.ClassDef):
                            class_complexity = self._calculate_class_cognitive_complexity(node)
                            cognitive_analysis["class_complexities"].append({
                                "class": node.name,
                                "file": str(file_path),
                                "complexity": class_complexity,
                                "method_count": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                                "avg_method_complexity": self._calculate_avg_method_complexity(node)
                            })
                            
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
                
        # Calculate distribution
        for func in cognitive_analysis["function_complexities"]:
            category = func["category"]
            cognitive_analysis["complexity_distribution"][category] += 1
            
        return cognitive_analysis
    
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """
        Calculate cognitive complexity for a node
        Based on SonarSource's Cognitive Complexity metric
        """
        complexity = 0
        nesting_level = 0
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting_level = 0
                
            def visit_If(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_While(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_For(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_ExceptHandler(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_BoolOp(self, node):
                # Each additional boolean operator adds complexity
                self.complexity += len(node.values) - 1
                self.generic_visit(node)
                
            def visit_Lambda(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_ListComp(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_DictComp(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_SetComp(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_GeneratorExp(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_Break(self, node):
                self.complexity += 1
                
            def visit_Continue(self, node):
                self.complexity += 1
                
            def visit_Return(self, node):
                # Multiple returns add complexity
                self.complexity += 1
                
        visitor = ComplexityVisitor()
        visitor.visit(node)
        return visitor.complexity
    
    def _calculate_halstead_metrics(self) -> Dict[str, Any]:
        """
        Calculate Halstead complexity metrics
        Measures program vocabulary, length, difficulty, effort, and time
        """
        halstead_results = {
            "metrics_by_function": [],
            "metrics_by_module": [],
            "aggregate_metrics": {},
            "effort_distribution": defaultdict(list),
            "vocabulary_analysis": {}
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                    
                tree = ast.parse(source)
                operators, operands = self._extract_operators_operands(tree)
                
                # Calculate Halstead metrics
                n1 = len(set(operators))  # Unique operators
                n2 = len(set(operands))   # Unique operands
                N1 = len(operators)       # Total operators
                N2 = len(operands)        # Total operands
                
                if n1 > 0 and n2 > 0 and N1 + N2 > 0:
                    vocabulary = n1 + n2
                    length = N1 + N2
                    volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
                    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
                    effort = difficulty * volume
                    time = effort / 18  # Stroud number
                    bugs = volume / 3000  # Empirical bug estimate
                    
                    module_metrics = {
                        "file": str(file_path),
                        "vocabulary": vocabulary,
                        "length": length,
                        "volume": volume,
                        "difficulty": difficulty,
                        "effort": effort,
                        "time_seconds": time,
                        "estimated_bugs": bugs,
                        "unique_operators": n1,
                        "unique_operands": n2,
                        "total_operators": N1,
                        "total_operands": N2
                    }
                    
                    halstead_results["metrics_by_module"].append(module_metrics)
                    
                    # Analyze functions individually
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_ops, func_opnds = self._extract_operators_operands(node)
                            if func_ops and func_opnds:
                                func_metrics = self._calculate_halstead_for_function(
                                    func_ops, func_opnds, node.name, str(file_path)
                                )
                                halstead_results["metrics_by_function"].append(func_metrics)
                                
            except Exception as e:
                self.logger.error(f"Error calculating Halstead metrics for {file_path}: {e}")
                
        # Calculate aggregate metrics
        if halstead_results["metrics_by_module"]:
            halstead_results["aggregate_metrics"] = {
                "total_vocabulary": sum(m["vocabulary"] for m in halstead_results["metrics_by_module"]),
                "total_length": sum(m["length"] for m in halstead_results["metrics_by_module"]),
                "total_volume": sum(m["volume"] for m in halstead_results["metrics_by_module"]),
                "avg_difficulty": sum(m["difficulty"] for m in halstead_results["metrics_by_module"]) / len(halstead_results["metrics_by_module"]),
                "total_effort": sum(m["effort"] for m in halstead_results["metrics_by_module"]),
                "total_time_hours": sum(m["time_seconds"] for m in halstead_results["metrics_by_module"]) / 3600,
                "total_estimated_bugs": sum(m["estimated_bugs"] for m in halstead_results["metrics_by_module"])
            }
            
        return halstead_results
    
    def _analyze_readability(self) -> Dict[str, Any]:
        """
        Analyze code readability using multiple factors
        """
        readability_analysis = {
            "readability_scores": [],
            "naming_analysis": self._analyze_comprehensive_naming(),
            "comment_density": self._analyze_comment_density(),
            "line_length_analysis": self._analyze_line_lengths(),
            "whitespace_usage": self._analyze_whitespace_usage(),
            "consistency_metrics": self._analyze_consistency_metrics(),
            "readability_issues": [],
            "improvement_suggestions": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Calculate various readability metrics
                readability_score = self._calculate_readability_score(lines, file_path)
                readability_analysis["readability_scores"].append({
                    "file": str(file_path),
                    "score": readability_score,
                    "grade": self._grade_readability(readability_score),
                    "factors": self._analyze_readability_factors(lines)
                })
                
                # Identify specific readability issues
                issues = self._identify_readability_issues(lines, file_path)
                readability_analysis["readability_issues"].extend(issues)
                
            except Exception as e:
                self.logger.error(f"Error analyzing readability for {file_path}: {e}")
                
        return readability_analysis
    
    def _calculate_maintainability_index(self) -> Dict[str, Any]:
        """
        Calculate Maintainability Index (MI)
        MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC)
        where V = Halstead Volume, CC = Cyclomatic Complexity, LOC = Lines of Code
        """
        maintainability_results = {
            "index_by_function": [],
            "index_by_module": [],
            "overall_index": 0,
            "maintainability_distribution": defaultdict(int),
            "refactoring_priorities": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        loc = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
                        
                    # Calculate module-level MI
                    module_cc = self._calculate_module_cyclomatic_complexity(tree)
                    operators, operands = self._extract_operators_operands(tree)
                    volume = self._calculate_halstead_volume(operators, operands)
                    
                    if loc > 0 and volume > 0:
                        mi = 171 - 5.2 * math.log(volume) - 0.23 * module_cc - 16.2 * math.log(loc)
                        mi = max(0, min(100, mi))  # Normalize to 0-100
                        
                        maintainability_results["index_by_module"].append({
                            "file": str(file_path),
                            "index": mi,
                            "grade": self._grade_maintainability(mi),
                            "loc": loc,
                            "cyclomatic_complexity": module_cc,
                            "halstead_volume": volume
                        })
                        
                        if mi < 50:  # Low maintainability
                            maintainability_results["refactoring_priorities"].append({
                                "file": str(file_path),
                                "index": mi,
                                "reason": self._analyze_low_maintainability_reason(mi, module_cc, volume, loc)
                            })
                            
                    # Calculate function-level MI
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_mi = self._calculate_function_maintainability(node, file_path)
                            if func_mi:
                                maintainability_results["index_by_function"].append(func_mi)
                                
            except Exception as e:
                self.logger.error(f"Error calculating MI for {file_path}: {e}")
                
        # Calculate overall maintainability index
        if maintainability_results["index_by_module"]:
            maintainability_results["overall_index"] = sum(
                m["index"] for m in maintainability_results["index_by_module"]
            ) / len(maintainability_results["index_by_module"])
            
        # Distribute by grade
        for module in maintainability_results["index_by_module"]:
            grade = module["grade"]
            maintainability_results["maintainability_distribution"][grade] += 1
            
        return maintainability_results
    
    def _analyze_nesting_depth(self) -> Dict[str, Any]:
        """
        Analyze nesting depth and its impact on cognitive load
        """
        nesting_analysis = {
            "max_nesting_depths": [],
            "avg_nesting_depth": 0,
            "deep_nesting_locations": [],
            "nesting_distribution": defaultdict(int),
            "nesting_patterns": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            max_depth = self._calculate_max_nesting_depth(node)
                            nesting_analysis["max_nesting_depths"].append({
                                "name": node.name,
                                "file": str(file_path),
                                "max_depth": max_depth,
                                "type": type(node).__name__
                            })
                            
                            if max_depth > 4:  # Deep nesting threshold
                                nesting_analysis["deep_nesting_locations"].append({
                                    "name": node.name,
                                    "file": str(file_path),
                                    "depth": max_depth,
                                    "cognitive_impact": self._assess_nesting_impact(max_depth)
                                })
                                
                            nesting_analysis["nesting_distribution"][max_depth] += 1
                            
            except Exception as e:
                self.logger.error(f"Error analyzing nesting for {file_path}: {e}")
                
        # Calculate average
        if nesting_analysis["max_nesting_depths"]:
            nesting_analysis["avg_nesting_depth"] = sum(
                n["max_depth"] for n in nesting_analysis["max_nesting_depths"]
            ) / len(nesting_analysis["max_nesting_depths"])
            
        return nesting_analysis
    
    def _analyze_naming_quality(self) -> Dict[str, Any]:
        """
        Analyze variable and function naming quality
        """
        naming_analysis = {
            "naming_conventions": self._detect_naming_conventions(),
            "naming_consistency": self._check_naming_consistency(),
            "naming_clarity": self._assess_naming_clarity(),
            "abbreviation_usage": self._detect_abbreviations(),
            "naming_violations": [],
            "improvement_suggestions": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Analyze variable names
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name):
                            quality = self._assess_name_quality(node.id)
                            if quality["score"] < 0.5:
                                naming_analysis["naming_violations"].append({
                                    "name": node.id,
                                    "file": str(file_path),
                                    "type": "variable",
                                    "issues": quality["issues"],
                                    "suggestion": self._suggest_better_name(node.id)
                                })
                                
                        elif isinstance(node, ast.FunctionDef):
                            quality = self._assess_name_quality(node.name)
                            if quality["score"] < 0.5:
                                naming_analysis["naming_violations"].append({
                                    "name": node.name,
                                    "file": str(file_path),
                                    "type": "function",
                                    "issues": quality["issues"],
                                    "suggestion": self._suggest_better_name(node.name)
                                })
                                
            except Exception as e:
                self.logger.error(f"Error analyzing naming for {file_path}: {e}")
                
        return naming_analysis
    
    def _analyze_abstraction_levels(self) -> Dict[str, Any]:
        """
        Analyze abstraction levels and their appropriateness
        """
        abstraction_analysis = {
            "abstraction_levels": [],
            "abstraction_violations": [],
            "leaky_abstractions": [],
            "abstraction_balance": {},
            "refactoring_opportunities": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Analyze classes for abstraction
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            abstraction_level = self._assess_abstraction_level(node)
                            abstraction_analysis["abstraction_levels"].append({
                                "class": node.name,
                                "file": str(file_path),
                                "level": abstraction_level["level"],
                                "appropriateness": abstraction_level["appropriateness"],
                                "cohesion": abstraction_level["cohesion"],
                                "coupling": abstraction_level["coupling"]
                            })
                            
                            # Check for abstraction violations
                            if abstraction_level["violations"]:
                                abstraction_analysis["abstraction_violations"].extend(
                                    abstraction_level["violations"]
                                )
                                
                            # Detect leaky abstractions
                            if abstraction_level["leaks"]:
                                abstraction_analysis["leaky_abstractions"].extend(
                                    abstraction_level["leaks"]
                                )
                                
            except Exception as e:
                self.logger.error(f"Error analyzing abstractions for {file_path}: {e}")
                
        return abstraction_analysis
    
    def _analyze_pattern_consistency(self) -> Dict[str, Any]:
        """
        Analyze consistency in coding patterns
        """
        consistency_analysis = {
            "pattern_usage": defaultdict(int),
            "inconsistencies": [],
            "pattern_violations": [],
            "consistency_score": 0,
            "recommendations": []
        }
        
        patterns_found = defaultdict(list)
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Detect various patterns
                    for node in ast.walk(tree):
                        pattern = self._identify_pattern(node)
                        if pattern:
                            patterns_found[pattern["type"]].append({
                                "file": str(file_path),
                                "pattern": pattern["name"],
                                "variation": pattern.get("variation", "standard")
                            })
                            
            except Exception as e:
                self.logger.error(f"Error analyzing patterns for {file_path}: {e}")
                
        # Analyze consistency
        for pattern_type, instances in patterns_found.items():
            variations = defaultdict(int)
            for instance in instances:
                variations[instance["variation"]] += 1
                
            if len(variations) > 1:
                # Inconsistent pattern usage
                consistency_analysis["inconsistencies"].append({
                    "pattern_type": pattern_type,
                    "variations": dict(variations),
                    "recommendation": self._recommend_pattern_standardization(pattern_type, variations)
                })
                
        # Calculate consistency score
        if patterns_found:
            total_patterns = sum(len(instances) for instances in patterns_found.values())
            consistent_patterns = sum(
                len(instances) for pattern_type, instances in patterns_found.items()
                if len(set(i["variation"] for i in instances)) == 1
            )
            consistency_analysis["consistency_score"] = consistent_patterns / total_patterns if total_patterns > 0 else 0
            
        return consistency_analysis
    
    def _analyze_mental_models(self) -> Dict[str, Any]:
        """
        Analyze mental models required to understand the code
        """
        mental_model_analysis = {
            "model_complexity": [],
            "conceptual_dependencies": [],
            "abstraction_mismatches": [],
            "cognitive_distance": {},
            "learning_dependencies": []
        }
        
        # Build dependency graph for mental models
        concept_graph = defaultdict(set)
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Extract concepts and their relationships
                    concepts = self._extract_concepts(tree)
                    for concept in concepts:
                        mental_model_analysis["model_complexity"].append({
                            "concept": concept["name"],
                            "file": str(file_path),
                            "complexity": concept["complexity"],
                            "dependencies": concept["dependencies"],
                            "cognitive_load": concept["cognitive_load"]
                        })
                        
                        # Build concept graph
                        for dep in concept["dependencies"]:
                            concept_graph[concept["name"]].add(dep)
                            
            except Exception as e:
                self.logger.error(f"Error analyzing mental models for {file_path}: {e}")
                
        # Calculate cognitive distance between concepts
        mental_model_analysis["cognitive_distance"] = self._calculate_cognitive_distance(concept_graph)
        
        # Identify learning dependencies
        mental_model_analysis["learning_dependencies"] = self._identify_learning_path(concept_graph)
        
        return mental_model_analysis
    
    def _analyze_code_shape(self) -> Dict[str, Any]:
        """
        Analyze the visual shape of code and its impact on readability
        """
        shape_analysis = {
            "shape_metrics": [],
            "arrow_antipattern": [],
            "pyramid_of_doom": [],
            "shape_distribution": defaultdict(int),
            "visual_complexity": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Analyze indentation patterns
                indentation_pattern = self._analyze_indentation_pattern(lines)
                
                # Detect arrow anti-pattern
                arrow_patterns = self._detect_arrow_antipattern(lines)
                if arrow_patterns:
                    shape_analysis["arrow_antipattern"].extend(arrow_patterns)
                    
                # Detect pyramid of doom
                pyramid_patterns = self._detect_pyramid_of_doom(lines)
                if pyramid_patterns:
                    shape_analysis["pyramid_of_doom"].extend(pyramid_patterns)
                    
                # Calculate visual complexity
                visual_complexity = self._calculate_visual_complexity(lines)
                shape_analysis["visual_complexity"].append({
                    "file": str(file_path),
                    "complexity": visual_complexity,
                    "indentation_variance": indentation_pattern["variance"],
                    "max_indentation": indentation_pattern["max_depth"],
                    "shape_category": self._categorize_code_shape(indentation_pattern)
                })
                
            except Exception as e:
                self.logger.error(f"Error analyzing code shape for {file_path}: {e}")
                
        return shape_analysis
    
    def _identify_cognitive_hotspots(self) -> Dict[str, Any]:
        """
        Identify areas of code with highest cognitive load
        """
        hotspots = {
            "critical_hotspots": [],
            "hotspot_clusters": [],
            "remediation_priority": [],
            "cognitive_debt": 0
        }
        
        all_metrics = []
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            metrics = self._calculate_comprehensive_metrics(node, file_path)
                            all_metrics.append(metrics)
                            
                            if metrics["cognitive_load_score"] > 75:  # High cognitive load
                                hotspots["critical_hotspots"].append({
                                    "name": node.name,
                                    "file": str(file_path),
                                    "type": type(node).__name__,
                                    "cognitive_load": metrics["cognitive_load_score"],
                                    "contributing_factors": metrics["contributing_factors"],
                                    "estimated_refactoring_effort": metrics["refactoring_effort"]
                                })
                                
            except Exception as e:
                self.logger.error(f"Error identifying hotspots for {file_path}: {e}")
                
        # Identify clusters of hotspots
        hotspots["hotspot_clusters"] = self._identify_hotspot_clusters(all_metrics)
        
        # Calculate total cognitive debt
        hotspots["cognitive_debt"] = sum(
            h["estimated_refactoring_effort"] for h in hotspots["critical_hotspots"]
        )
        
        # Prioritize remediation
        hotspots["remediation_priority"] = sorted(
            hotspots["critical_hotspots"],
            key=lambda x: x["cognitive_load"] * x["estimated_refactoring_effort"],
            reverse=True
        )[:10]  # Top 10 priorities
        
        return hotspots
    
    def _identify_comprehension_barriers(self) -> Dict[str, Any]:
        """
        Identify specific barriers to code comprehension
        """
        barriers = {
            "implicit_behaviors": [],
            "hidden_dependencies": [],
            "unclear_contracts": [],
            "surprising_behaviors": [],
            "knowledge_assumptions": [],
            "barrier_severity": defaultdict(int)
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Detect implicit behaviors
                    implicit = self._detect_implicit_behaviors(tree, file_path)
                    barriers["implicit_behaviors"].extend(implicit)
                    
                    # Find hidden dependencies
                    hidden = self._detect_hidden_dependencies(tree, file_path)
                    barriers["hidden_dependencies"].extend(hidden)
                    
                    # Identify unclear contracts
                    unclear = self._detect_unclear_contracts(tree, file_path)
                    barriers["unclear_contracts"].extend(unclear)
                    
                    # Find surprising behaviors
                    surprising = self._detect_surprising_behaviors(tree, file_path)
                    barriers["surprising_behaviors"].extend(surprising)
                    
                    # Detect knowledge assumptions
                    assumptions = self._detect_knowledge_assumptions(tree, file_path)
                    barriers["knowledge_assumptions"].extend(assumptions)
                    
            except Exception as e:
                self.logger.error(f"Error identifying barriers for {file_path}: {e}")
                
        # Categorize barrier severity
        for barrier_type in ["implicit_behaviors", "hidden_dependencies", "unclear_contracts", 
                            "surprising_behaviors", "knowledge_assumptions"]:
            for barrier in barriers[barrier_type]:
                severity = barrier.get("severity", "medium")
                barriers["barrier_severity"][severity] += 1
                
        return barriers
    
    def _estimate_learning_curve(self) -> Dict[str, Any]:
        """
        Estimate the learning curve for new developers
        """
        learning_curve = {
            "onboarding_time_estimate": {},
            "concept_complexity_levels": [],
            "prerequisite_knowledge": [],
            "learning_path": [],
            "documentation_coverage": {},
            "mentorship_requirements": []
        }
        
        # Analyze codebase complexity
        complexity_factors = {
            "domain_complexity": self._assess_domain_complexity(),
            "technical_complexity": self._assess_technical_complexity(),
            "architectural_complexity": self._assess_architectural_complexity(),
            "tooling_complexity": self._assess_tooling_complexity()
        }
        
        # Estimate onboarding time based on complexity
        total_complexity = sum(complexity_factors.values())
        learning_curve["onboarding_time_estimate"] = {
            "junior_developer_days": self._estimate_onboarding_time(total_complexity, "junior"),
            "mid_developer_days": self._estimate_onboarding_time(total_complexity, "mid"),
            "senior_developer_days": self._estimate_onboarding_time(total_complexity, "senior"),
            "complexity_breakdown": complexity_factors
        }
        
        # Build learning path
        learning_curve["learning_path"] = self._build_optimal_learning_path()
        
        # Assess documentation coverage
        learning_curve["documentation_coverage"] = self._assess_documentation_coverage()
        
        # Identify mentorship requirements
        learning_curve["mentorship_requirements"] = self._identify_mentorship_needs(complexity_factors)
        
        return learning_curve
    
    def _generate_refactoring_suggestions(self) -> Dict[str, Any]:
        """
        Generate specific refactoring suggestions to reduce cognitive load
        """
        suggestions = {
            "immediate_actions": [],
            "short_term_improvements": [],
            "long_term_refactoring": [],
            "estimated_impact": {},
            "refactoring_patterns": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            complexity = self._calculate_cognitive_complexity(node)
                            
                            if complexity > self.complexity_thresholds["very_high"]:
                                # Immediate action needed
                                suggestions["immediate_actions"].append({
                                    "function": node.name,
                                    "file": str(file_path),
                                    "current_complexity": complexity,
                                    "suggestion": "Extract method to reduce complexity",
                                    "specific_actions": self._generate_extraction_suggestions(node),
                                    "estimated_new_complexity": complexity * 0.5
                                })
                                
                            elif complexity > self.complexity_thresholds["high"]:
                                # Short-term improvement
                                suggestions["short_term_improvements"].append({
                                    "function": node.name,
                                    "file": str(file_path),
                                    "current_complexity": complexity,
                                    "suggestion": "Simplify control flow",
                                    "specific_actions": self._generate_simplification_suggestions(node)
                                })
                                
                        elif isinstance(node, ast.ClassDef):
                            # Analyze class for refactoring
                            class_suggestions = self._analyze_class_refactoring(node, file_path)
                            if class_suggestions:
                                suggestions["long_term_refactoring"].extend(class_suggestions)
                                
            except Exception as e:
                self.logger.error(f"Error generating suggestions for {file_path}: {e}")
                
        # Identify refactoring patterns
        suggestions["refactoring_patterns"] = self._identify_refactoring_patterns(suggestions)
        
        # Estimate overall impact
        suggestions["estimated_impact"] = self._estimate_refactoring_impact(suggestions)
        
        return suggestions
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of cognitive load analysis
        """
        summary = {
            "overall_cognitive_health": "unknown",
            "key_metrics": {},
            "top_issues": [],
            "improvement_potential": 0,
            "recommended_actions": [],
            "team_impact": {}
        }
        
        # Calculate overall cognitive health score
        health_score = self._calculate_cognitive_health_score()
        summary["overall_cognitive_health"] = self._grade_cognitive_health(health_score)
        
        # Compile key metrics
        summary["key_metrics"] = {
            "avg_cognitive_complexity": self._calculate_average_cognitive_complexity(),
            "maintainability_index": self._get_overall_maintainability_index(),
            "readability_score": self._get_overall_readability_score(),
            "consistency_score": self._get_consistency_score(),
            "cognitive_debt_hours": self._calculate_cognitive_debt_hours()
        }
        
        # Identify top issues
        summary["top_issues"] = self._identify_top_cognitive_issues()
        
        # Calculate improvement potential
        summary["improvement_potential"] = self._calculate_improvement_potential()
        
        # Generate recommended actions
        summary["recommended_actions"] = self._generate_action_plan()
        
        # Assess team impact
        summary["team_impact"] = {
            "productivity_impact": self._assess_productivity_impact(),
            "onboarding_difficulty": self._assess_onboarding_difficulty(),
            "maintenance_burden": self._assess_maintenance_burden(),
            "bug_risk": self._assess_bug_risk_from_complexity()
        }
        
        return summary
    
    # Helper methods
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate traditional cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _categorize_complexity(self, complexity: int) -> str:
        """Categorize complexity level"""
        if complexity <= self.complexity_thresholds["low"]:
            return "low"
        elif complexity <= self.complexity_thresholds["medium"]:
            return "medium"
        elif complexity <= self.complexity_thresholds["high"]:
            return "high"
        else:
            return "very_high"
    
    def _extract_operators_operands(self, node: ast.AST) -> Tuple[List[str], List[str]]:
        """Extract operators and operands for Halstead metrics"""
        operators = []
        operands = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp):
                operators.append(type(child.op).__name__)
            elif isinstance(child, ast.UnaryOp):
                operators.append(type(child.op).__name__)
            elif isinstance(child, ast.Compare):
                for op in child.ops:
                    operators.append(type(op).__name__)
            elif isinstance(child, ast.Name):
                operands.append(child.id)
            elif isinstance(child, ast.Constant):
                operands.append(str(child.value))
                
        return operators, operands
    
    def _calculate_halstead_volume(self, operators: List[str], operands: List[str]) -> float:
        """Calculate Halstead volume metric"""
        n1 = len(set(operators))
        n2 = len(set(operands))
        N1 = len(operators)
        N2 = len(operands)
        
        if n1 + n2 > 0 and N1 + N2 > 0:
            vocabulary = n1 + n2
            length = N1 + N2
            return length * math.log2(vocabulary) if vocabulary > 0 else 0
        return 0
    
    def _calculate_max_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth in a node"""
        max_depth = 0
        
        class DepthVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_depth = 0
                self.max_depth = 0
                
            def visit_If(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit_While(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit_For(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit_With(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit_Try(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
        visitor = DepthVisitor()
        visitor.visit(node)
        return visitor.max_depth
    
    def _assess_name_quality(self, name: str) -> Dict[str, Any]:
        """Assess the quality of a variable or function name"""
        quality = {
            "score": 1.0,
            "issues": []
        }
        
        # Check length
        if len(name) < 2:
            quality["score"] *= 0.5
            quality["issues"].append("Too short")
        elif len(name) > 30:
            quality["score"] *= 0.7
            quality["issues"].append("Too long")
            
        # Check for single letters (except common ones like i, j for loops)
        if len(name) == 1 and name not in ['i', 'j', 'k', 'n', 'x', 'y', 'z']:
            quality["score"] *= 0.3
            quality["issues"].append("Single letter name")
            
        # Check for unclear abbreviations
        common_abbrevs = ['num', 'val', 'idx', 'cnt', 'tmp', 'src', 'dst', 'msg']
        if any(abbrev in name.lower() for abbrev in common_abbrevs):
            quality["score"] *= 0.8
            quality["issues"].append("Contains abbreviation")
            
        # Check for meaningful words
        if not any(c.isupper() or c == '_' for c in name[1:]):
            quality["score"] *= 0.9
            quality["issues"].append("No word separation")
            
        return quality
    
    def _suggest_better_name(self, name: str) -> str:
        """Suggest a better name based on common patterns"""
        suggestions = {
            'tmp': 'temporary_value',
            'val': 'value',
            'num': 'number',
            'cnt': 'count',
            'idx': 'index',
            'src': 'source',
            'dst': 'destination',
            'msg': 'message'
        }
        
        for abbrev, full in suggestions.items():
            if abbrev in name.lower():
                return name.lower().replace(abbrev, full)
                
        return f"descriptive_name_for_{name}"
    
    def _grade_readability(self, score: float) -> str:
        """Grade readability score"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        elif score >= 40:
            return "poor"
        else:
            return "very_poor"
    
    def _grade_maintainability(self, index: float) -> str:
        """Grade maintainability index"""
        if index >= 85:
            return "highly_maintainable"
        elif index >= 65:
            return "moderately_maintainable"
        elif index >= 50:
            return "difficult_to_maintain"
        else:
            return "unmaintainable"