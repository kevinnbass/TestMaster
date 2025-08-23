#!/usr/bin/env python3
"""
🏗️ MODULE: Quality Analyzer - Extracted from Enhanced Intelligence Linkage
==========================================================================

📋 PURPOSE:
    Comprehensive code quality analysis engine providing maintainability metrics,
    complexity analysis, technical debt assessment, and quality correlation analysis.

🎯 CORE FUNCTIONALITY:
    • Cyclomatic complexity calculation and analysis
    • Maintainability index assessment with level classification
    • Technical debt detection and scoring
    • Quality linkage correlation analysis

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-23] | Agent Z | 🔧 STEELCLAD
   └─ Goal: Extract quality analysis from enhanced_intelligence_linkage.py
   └─ Changes: Modularized quality analysis with ~160 lines of focused functionality
   └─ Impact: Reduces main intelligence linkage size while maintaining quality analysis

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent Z (STEELCLAD extraction)
🔧 Language: Python
📦 Dependencies: re, math, pathlib
🎯 Integration Points: EnhancedLinkageAnalyzer class
⚡ Performance Notes: Optimized for large-scale quality assessment
🔒 Security Notes: Safe code analysis with error handling
"""

import re
import math
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class QualityAnalyzer:
    """Comprehensive code quality analysis engine."""
    
    def __init__(self):
        self.quality_thresholds = self._load_quality_thresholds()
        self.complexity_patterns = self._load_complexity_patterns()
        self.debt_patterns = self._load_debt_patterns()
        self.maintainability_rules = self._load_maintainability_rules()
    
    def _load_quality_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load quality assessment thresholds."""
        return {
            "complexity": {
                "low": 10,
                "medium": 20,
                "high": 30,
                "very_high": 50
            },
            "maintainability": {
                "excellent": 85,
                "good": 70,
                "fair": 50,
                "poor": 25
            },
            "debt": {
                "low": 5,
                "medium": 15,
                "high": 30,
                "critical": 50
            },
            "lines_of_code": {
                "small": 50,
                "medium": 200,
                "large": 500,
                "very_large": 1000
            }
        }
    
    def _load_complexity_patterns(self) -> Dict[str, List[str]]:
        """Load complexity calculation patterns."""
        return {
            "decision_points": [
                r'\bif\b', r'\belif\b', r'\belse\b',
                r'\bfor\b', r'\bwhile\b',
                r'\band\b', r'\bor\b',
                r'\btry\b', r'\bexcept\b',
                r'\bcase\b', r'\bswitch\b'
            ],
            "loop_patterns": [
                r'\bfor\b.*:', r'\bwhile\b.*:',
                r'list\s*comprehension', r'dict\s*comprehension'
            ],
            "nesting_indicators": [
                r'\n\s{8,}', r'\n\s{12,}', r'\n\s{16,}', r'\n\s{20,}'
            ]
        }
    
    def _load_debt_patterns(self) -> Dict[str, List[str]]:
        """Load technical debt detection patterns."""
        return {
            "todo_comments": [
                r"#.*TODO", r"#.*FIXME", r"#.*HACK", r"#.*XXX",
                r"#.*BUG", r"#.*TEMP", r"#.*KLUDGE"
            ],
            "code_smells": [
                r"def .{50,}", r"class .{50,}",  # Long names
                r"def \w+\([^)]{100,}", r"lambda .{50,}"  # Long parameter lists/lambdas
            ],
            "magic_numbers": [
                r"\b\d{4,}\b", r"\b\d+\.\d{4,}\b"
            ],
            "duplication_indicators": [
                r"(.{20,})\1", r"def \w+.*?def \1"  # Potential duplication patterns
            ],
            "anti_patterns": [
                r"global \w+", r"exec\(", r"eval\(",
                r"import \*", r"__import__"
            ]
        }
    
    def _load_maintainability_rules(self) -> Dict[str, Any]:
        """Load maintainability assessment rules."""
        return {
            "documentation_weight": 0.2,
            "complexity_weight": 0.3,
            "structure_weight": 0.3,
            "naming_weight": 0.2,
            "max_function_length": 50,
            "max_class_length": 300,
            "max_nesting_depth": 4
        }
    
    def analyze_quality_dimensions(self, python_files: List[Path], base_dir: str) -> Dict[str, Any]:
        """Comprehensive quality analysis of codebase."""
        quality_results = {
            "complexity_scores": {},
            "maintainability_index": {},
            "technical_debt": {},
            "quality_linkage_correlation": {},
            "quality_metrics": {}
        }
        
        base_path = Path(base_dir)
        total_files = len(python_files)
        
        print(f"Quality Analysis: Processing {total_files} files...")
        
        for i, py_file in enumerate(python_files):
            if i % 100 == 0:  # Progress update every 100 files
                print(f"  Quality analysis progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            try:
                relative_path = str(py_file.relative_to(base_path))
                
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Complexity analysis
                complexity = self.calculate_complexity(content)
                quality_results["complexity_scores"][relative_path] = complexity
                
                # Maintainability index
                maintainability = self.calculate_maintainability_index(content, complexity)
                quality_results["maintainability_index"][relative_path] = maintainability
                
                # Technical debt assessment
                debt = self.assess_technical_debt(content)
                quality_results["technical_debt"][relative_path] = debt
                
            except Exception as e:
                print(f"  Error processing {py_file}: {e}")
                continue
        
        # Quality linkage correlation
        quality_results["quality_linkage_correlation"] = self.analyze_quality_linkage_correlation(
            quality_results["complexity_scores"],
            quality_results["maintainability_index"],
            quality_results["technical_debt"]
        )
        
        # Calculate quality metrics
        quality_results["quality_metrics"] = self.calculate_quality_metrics(
            quality_results
        )
        
        print("Quality Analysis: Complete!")
        return quality_results
    
    def calculate_complexity(self, content: str) -> Dict[str, Any]:
        """Calculate comprehensive complexity metrics."""
        lines_of_code = len([line for line in content.splitlines() if line.strip()])
        total_complexity = 1  # Base complexity
        
        # Count decision points
        decision_complexity = 0
        for pattern in self.complexity_patterns["decision_points"]:
            try:
                decision_complexity += len(re.findall(pattern, content))
            except re.error:
                continue
        
        total_complexity += decision_complexity
        
        # Analyze nesting depth
        max_nesting = self._calculate_nesting_depth(content)
        nesting_complexity = max_nesting * 2 if max_nesting > 3 else 0
        
        # Function and class complexity
        function_complexity = self._calculate_function_complexity(content)
        class_complexity = self._calculate_class_complexity(content)
        
        # Cognitive complexity (additional metric)
        cognitive_complexity = self._calculate_cognitive_complexity(content)
        
        return {
            "cyclomatic_complexity": total_complexity,
            "decision_complexity": decision_complexity,
            "nesting_complexity": nesting_complexity,
            "cognitive_complexity": cognitive_complexity,
            "lines_of_code": lines_of_code,
            "complexity_density": total_complexity / max(lines_of_code, 1),
            "function_complexity": function_complexity,
            "class_complexity": class_complexity,
            "max_nesting_depth": max_nesting,
            "complexity_level": self._get_complexity_level(total_complexity),
            "size_category": self._get_size_category(lines_of_code)
        }
    
    def calculate_maintainability_index(self, content: str, complexity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate maintainability index with comprehensive assessment."""
        loc = complexity["lines_of_code"]
        cc = complexity["cyclomatic_complexity"]
        
        if loc == 0:
            return {
                "maintainability_index": 100,
                "maintainability_level": "excellent",
                "factors": {}
            }
        
        try:
            # Enhanced maintainability index calculation
            # MI = 171 - 5.2 * log(avgV) - 0.23 * avgCC - 16.2 * log(avgLOC)
            # Where avgV is average volume (simplified as LOC here)
            
            volume_factor = 5.2 * math.log(max(loc, 1))
            complexity_factor = 0.23 * cc
            size_factor = 16.2 * math.log(max(loc, 1))
            
            base_mi = 171 - volume_factor - complexity_factor - size_factor
            
            # Adjust for documentation
            doc_factor = self._calculate_documentation_factor(content)
            structure_factor = self._calculate_structure_factor(content)
            naming_factor = self._calculate_naming_factor(content)
            
            # Apply weights
            rules = self.maintainability_rules
            adjusted_mi = base_mi + (
                doc_factor * rules["documentation_weight"] * 20 +
                structure_factor * rules["structure_weight"] * 15 +
                naming_factor * rules["naming_weight"] * 10
            )
            
            # Clamp between 0-100
            mi = max(0, min(100, adjusted_mi))
            
        except (ValueError, OverflowError, ZeroDivisionError):
            mi = 50  # Default value
        
        return {
            "maintainability_index": mi,
            "maintainability_level": self._get_maintainability_level(mi),
            "factors": {
                "documentation_factor": doc_factor,
                "structure_factor": structure_factor,
                "naming_factor": naming_factor,
                "complexity_impact": complexity_factor,
                "size_impact": size_factor
            },
            "recommendations": self._generate_maintainability_recommendations(mi, complexity)
        }
    
    def assess_technical_debt(self, content: str) -> Dict[str, Any]:
        """Comprehensive technical debt assessment."""
        debt_indicators = {}
        total_debt_score = 0
        
        # Count different types of technical debt
        for debt_type, patterns in self.debt_patterns.items():
            count = 0
            locations = []
            
            for pattern in patterns:
                try:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    count += len(matches)
                    
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        locations.append({
                            "line": line_number,
                            "text": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0)
                        })
                except re.error:
                    continue
            
            debt_indicators[debt_type] = {
                "count": count,
                "locations": locations[:5]  # Limit to first 5 locations
            }
            total_debt_score += count
        
        # Additional debt factors
        long_functions = self._count_long_functions(content)
        long_classes = self._count_long_classes(content)
        deep_nesting = self._count_deep_nesting(content)
        
        additional_debt = {
            "long_functions": long_functions,
            "long_classes": long_classes,
            "deep_nesting": deep_nesting
        }
        
        total_debt_score += sum(additional_debt.values())
        
        return {
            "debt_score": total_debt_score,
            "debt_indicators": debt_indicators,
            "additional_debt": additional_debt,
            "debt_level": self._get_debt_level(total_debt_score),
            "debt_density": total_debt_score / max(len(content.splitlines()), 1),
            "priority_areas": self._identify_priority_debt_areas(debt_indicators, additional_debt)
        }
    
    def analyze_quality_linkage_correlation(self, complexity_scores: Dict, 
                                          maintainability_scores: Dict, 
                                          debt_scores: Dict) -> Dict[str, Any]:
        """Analyze correlation between quality metrics and file linkages."""
        correlation_analysis = {
            "complexity_patterns": {},
            "maintainability_clusters": {},
            "debt_hotspots": [],
            "quality_trends": {}
        }
        
        # Analyze complexity patterns
        complexity_levels = defaultdict(list)
        for file_path, complexity_data in complexity_scores.items():
            level = complexity_data.get("complexity_level", "medium")
            complexity_levels[level].append(file_path)
        
        correlation_analysis["complexity_patterns"] = dict(complexity_levels)
        
        # Analyze maintainability clusters
        maintainability_levels = defaultdict(list)
        for file_path, maintainability_data in maintainability_scores.items():
            level = maintainability_data.get("maintainability_level", "fair")
            maintainability_levels[level].append(file_path)
        
        correlation_analysis["maintainability_clusters"] = dict(maintainability_levels)
        
        # Identify debt hotspots
        debt_hotspots = []
        for file_path, debt_data in debt_scores.items():
            debt_score = debt_data.get("debt_score", 0)
            if debt_score > 20:  # High debt threshold
                debt_hotspots.append({
                    "file": file_path,
                    "debt_score": debt_score,
                    "debt_level": debt_data.get("debt_level", "medium")
                })
        
        correlation_analysis["debt_hotspots"] = sorted(
            debt_hotspots, key=lambda x: x["debt_score"], reverse=True
        )[:10]  # Top 10 debt hotspots
        
        return correlation_analysis
    
    def calculate_quality_metrics(self, quality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics."""
        complexity_scores = quality_results.get("complexity_scores", {})
        maintainability_scores = quality_results.get("maintainability_index", {})
        debt_scores = quality_results.get("technical_debt", {})
        
        if not complexity_scores:
            return {"error": "No quality data available"}
        
        total_files = len(complexity_scores)
        
        # Average metrics
        avg_complexity = sum(
            data.get("cyclomatic_complexity", 0) for data in complexity_scores.values()
        ) / max(total_files, 1)
        
        avg_maintainability = sum(
            data.get("maintainability_index", 0) for data in maintainability_scores.values()
        ) / max(len(maintainability_scores), 1)
        
        avg_debt = sum(
            data.get("debt_score", 0) for data in debt_scores.values()
        ) / max(len(debt_scores), 1)
        
        # Quality distribution
        complexity_distribution = defaultdict(int)
        maintainability_distribution = defaultdict(int)
        debt_distribution = defaultdict(int)
        
        for data in complexity_scores.values():
            level = data.get("complexity_level", "medium")
            complexity_distribution[level] += 1
        
        for data in maintainability_scores.values():
            level = data.get("maintainability_level", "fair")
            maintainability_distribution[level] += 1
        
        for data in debt_scores.values():
            level = data.get("debt_level", "medium")
            debt_distribution[level] += 1
        
        return {
            "total_files_analyzed": total_files,
            "average_complexity": avg_complexity,
            "average_maintainability": avg_maintainability,
            "average_debt_score": avg_debt,
            "complexity_distribution": dict(complexity_distribution),
            "maintainability_distribution": dict(maintainability_distribution),
            "debt_distribution": dict(debt_distribution),
            "quality_grade": self._calculate_overall_quality_grade(
                avg_complexity, avg_maintainability, avg_debt
            ),
            "improvement_areas": self._identify_improvement_areas(quality_results)
        }
    
    def _calculate_nesting_depth(self, content: str) -> int:
        """Calculate maximum nesting depth."""
        lines = content.splitlines()
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Calculate indentation level
            indent_level = (len(line) - len(line.lstrip())) // 4
            
            if stripped.endswith(':') and not stripped.startswith(('"""', "'''")):
                current_depth = indent_level + 1
                max_depth = max(max_depth, current_depth)
        
        return max_depth
    
    def _calculate_function_complexity(self, content: str) -> Dict[str, Any]:
        """Calculate function-specific complexity metrics."""
        function_pattern = r'def\s+(\w+)\s*\([^)]*\):'
        functions = re.findall(function_pattern, content)
        
        return {
            "function_count": len(functions),
            "average_functions_per_100_loc": len(functions) / max(len(content.splitlines()) / 100, 1)
        }
    
    def _calculate_class_complexity(self, content: str) -> Dict[str, Any]:
        """Calculate class-specific complexity metrics."""
        class_pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
        classes = re.findall(class_pattern, content)
        
        return {
            "class_count": len(classes),
            "average_classes_per_100_loc": len(classes) / max(len(content.splitlines()) / 100, 1)
        }
    
    def _calculate_cognitive_complexity(self, content: str) -> int:
        """Calculate cognitive complexity (how hard code is to understand)."""
        cognitive_score = 0
        
        # Nested structures add to cognitive load
        nesting_depth = self._calculate_nesting_depth(content)
        cognitive_score += nesting_depth * 2
        
        # Complex boolean expressions
        complex_boolean = len(re.findall(r'and.*or|or.*and', content, re.IGNORECASE))
        cognitive_score += complex_boolean * 2
        
        # Exception handling complexity
        exception_blocks = len(re.findall(r'try:|except:|finally:', content))
        cognitive_score += exception_blocks
        
        return cognitive_score
    
    def _calculate_documentation_factor(self, content: str) -> float:
        """Calculate documentation quality factor (0-1)."""
        total_lines = len(content.splitlines())
        if total_lines == 0:
            return 0.5
        
        # Count docstrings and comments
        docstrings = len(re.findall(r'""".*?"""', content, re.DOTALL))
        docstrings += len(re.findall(r"'''.*?'''", content, re.DOTALL))
        comments = len(re.findall(r'#.*', content))
        
        doc_ratio = (docstrings * 10 + comments) / total_lines
        return min(1.0, doc_ratio)
    
    def _calculate_structure_factor(self, content: str) -> float:
        """Calculate code structure quality factor (0-1)."""
        # Function to class ratio
        functions = len(re.findall(r'def\s+\w+', content))
        classes = len(re.findall(r'class\s+\w+', content))
        
        # Balanced structure is good
        if functions == 0 and classes == 0:
            return 0.3
        
        if classes == 0:
            structure_score = 0.6  # Functions only
        elif functions == 0:
            structure_score = 0.4  # Classes only
        else:
            ratio = functions / (functions + classes)
            structure_score = 1.0 - abs(ratio - 0.7)  # Optimal is ~70% functions
        
        return max(0.0, min(1.0, structure_score))
    
    def _calculate_naming_factor(self, content: str) -> float:
        """Calculate naming quality factor (0-1)."""
        # Look for descriptive names
        descriptive_names = len(re.findall(r'\b[a-z_]{4,}\w*\b', content))
        total_identifiers = len(re.findall(r'\b[a-zA-Z_]\w*\b', content))
        
        if total_identifiers == 0:
            return 0.5
        
        naming_score = descriptive_names / total_identifiers
        return min(1.0, naming_score)
    
    def _count_long_functions(self, content: str) -> int:
        """Count functions longer than threshold."""
        max_length = self.maintainability_rules["max_function_length"]
        function_pattern = r'def\s+\w+.*?(?=\ndef|\nclass|\Z)'
        functions = re.findall(function_pattern, content, re.DOTALL)
        
        return len([f for f in functions if len(f.splitlines()) > max_length])
    
    def _count_long_classes(self, content: str) -> int:
        """Count classes longer than threshold."""
        max_length = self.maintainability_rules["max_class_length"]
        class_pattern = r'class\s+\w+.*?(?=\nclass|\ndef(?!\s+\w+.*:)|\Z)'
        classes = re.findall(class_pattern, content, re.DOTALL)
        
        return len([c for c in classes if len(c.splitlines()) > max_length])
    
    def _count_deep_nesting(self, content: str) -> int:
        """Count instances of deep nesting."""
        max_depth = self.maintainability_rules["max_nesting_depth"]
        deep_nesting = len(re.findall(rf'\n\s{{{max_depth * 4},}}', content))
        return deep_nesting
    
    def _get_complexity_level(self, complexity: int) -> str:
        """Get complexity level based on score."""
        thresholds = self.quality_thresholds["complexity"]
        if complexity <= thresholds["low"]:
            return "low"
        elif complexity <= thresholds["medium"]:
            return "medium"
        elif complexity <= thresholds["high"]:
            return "high"
        else:
            return "very_high"
    
    def _get_maintainability_level(self, mi: float) -> str:
        """Get maintainability level based on index."""
        thresholds = self.quality_thresholds["maintainability"]
        if mi >= thresholds["excellent"]:
            return "excellent"
        elif mi >= thresholds["good"]:
            return "good"
        elif mi >= thresholds["fair"]:
            return "fair"
        else:
            return "poor"
    
    def _get_debt_level(self, debt_score: int) -> str:
        """Get technical debt level based on score."""
        thresholds = self.quality_thresholds["debt"]
        if debt_score <= thresholds["low"]:
            return "low"
        elif debt_score <= thresholds["medium"]:
            return "medium"
        elif debt_score <= thresholds["high"]:
            return "high"
        else:
            return "critical"
    
    def _get_size_category(self, loc: int) -> str:
        """Get size category based on lines of code."""
        thresholds = self.quality_thresholds["lines_of_code"]
        if loc <= thresholds["small"]:
            return "small"
        elif loc <= thresholds["medium"]:
            return "medium"
        elif loc <= thresholds["large"]:
            return "large"
        else:
            return "very_large"
    
    def _generate_maintainability_recommendations(self, mi: float, complexity: Dict) -> List[str]:
        """Generate maintainability improvement recommendations."""
        recommendations = []
        
        if mi < 50:
            recommendations.append("Critical: Refactor for improved maintainability")
        elif mi < 70:
            recommendations.append("Consider refactoring to improve maintainability")
        
        if complexity.get("max_nesting_depth", 0) > 4:
            recommendations.append("Reduce nesting depth for better readability")
        
        if complexity.get("cyclomatic_complexity", 0) > 20:
            recommendations.append("Break down complex functions into smaller units")
        
        return recommendations
    
    def _identify_priority_debt_areas(self, debt_indicators: Dict, additional_debt: Dict) -> List[str]:
        """Identify priority areas for debt reduction."""
        priority_areas = []
        
        # Check each debt type
        for debt_type, data in debt_indicators.items():
            count = data.get("count", 0)
            if count > 5:  # High count threshold
                priority_areas.append(f"Address {debt_type} ({count} instances)")
        
        # Check additional debt
        if additional_debt.get("long_functions", 0) > 3:
            priority_areas.append("Refactor long functions for better maintainability")
        
        if additional_debt.get("deep_nesting", 0) > 5:
            priority_areas.append("Reduce deep nesting for improved readability")
        
        return priority_areas
    
    def _calculate_overall_quality_grade(self, avg_complexity: float, 
                                       avg_maintainability: float, avg_debt: float) -> str:
        """Calculate overall quality grade."""
        # Normalize scores (0-100)
        complexity_score = max(0, 100 - (avg_complexity * 2))  # Lower complexity is better
        maintainability_score = avg_maintainability  # Higher is better
        debt_score = max(0, 100 - (avg_debt * 2))  # Lower debt is better
        
        overall_score = (complexity_score + maintainability_score + debt_score) / 3
        
        if overall_score >= 85:
            return "A"
        elif overall_score >= 75:
            return "B"
        elif overall_score >= 65:
            return "C"
        elif overall_score >= 55:
            return "D"
        else:
            return "F"
    
    def _identify_improvement_areas(self, quality_results: Dict[str, Any]) -> List[str]:
        """Identify key areas for quality improvement."""
        improvement_areas = []
        
        complexity_scores = quality_results.get("complexity_scores", {})
        debt_scores = quality_results.get("technical_debt", {})
        
        # Check for high complexity files
        high_complexity_files = sum(
            1 for data in complexity_scores.values()
            if data.get("complexity_level") in ["high", "very_high"]
        )
        
        if high_complexity_files > len(complexity_scores) * 0.2:  # More than 20%
            improvement_areas.append("Reduce complexity in high-complexity files")
        
        # Check for high debt files
        high_debt_files = sum(
            1 for data in debt_scores.values()
            if data.get("debt_level") in ["high", "critical"]
        )
        
        if high_debt_files > len(debt_scores) * 0.1:  # More than 10%
            improvement_areas.append("Address technical debt in problematic files")
        
        return improvement_areas

def create_quality_analyzer() -> QualityAnalyzer:
    """Factory function to create a configured quality analyzer."""
    return QualityAnalyzer()