#!/usr/bin/env python3
"""
Agent B Phase 3: Hours 51-55 - Function Modularization Analysis
Identifies oversized functions and provides modularization recommendations.
"""

import ast
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class FunctionMetrics:
    """Metrics for a single function."""
    name: str
    file_path: str
    line_count: int
    complexity_score: int
    parameter_count: int
    return_statements: int
    nested_depth: int
    docstring_present: bool
    modularization_priority: str
    
@dataclass
class ModularizationRecommendation:
    """Recommendation for modularizing a function."""
    original_function: str
    file_path: str
    issue_type: str
    priority: str
    recommended_splits: List[str]
    new_function_names: List[str]
    rationale: str

class FunctionModularizationAnalyzer:
    """Analyzes functions for modularization opportunities."""
    
    def __init__(self, base_directory: str = "."):
        self.base_directory = base_directory
        self.function_metrics: List[FunctionMetrics] = []
        self.recommendations: List[ModularizationRecommendation] = []
        
        # Critical modules identified in Phase 2
        self.critical_modules = [
            "core/intelligence/__init__.py",
            "core/intelligence/testing/__init__.py", 
            "core/intelligence/analytics/__init__.py",
            "testmaster_orchestrator.py",
            "agentic_test_monitor.py",
            "config/__init__.py",
            "api/serializers.py",
            "api/validators.py",
            "workflow/__init__.py",
            "TestMaster/testmaster_orchestrator.py"
        ]
    
    def analyze_function_modularization(self) -> Dict:
        """Main analysis method for function modularization."""
        print("Starting Function Modularization Analysis...")
        print(f"Analyzing critical modules: {len(self.critical_modules)}")
        
        for module_path in self.critical_modules:
            self._analyze_module_functions(module_path)
        
        # Also analyze Python files in TestMaster directory
        self._analyze_testmaster_functions()
        
        self._generate_modularization_recommendations()
        
        results = {
            "analysis_metadata": {
                "analyzer": "Agent B - Function Modularization Analysis",
                "phase": "Hours 51-55",
                "timestamp": datetime.now().isoformat(),
                "functions_analyzed": len(self.function_metrics),
                "recommendations_generated": len(self.recommendations)
            },
            "function_metrics": [asdict(fm) for fm in self.function_metrics],
            "modularization_recommendations": [asdict(mr) for mr in self.recommendations],
            "summary": self._generate_summary()
        }
        
        return results
    
    def _analyze_module_functions(self, module_path: str):
        """Analyze functions in a specific module."""
        full_path = os.path.join(self.base_directory, module_path)
        
        if not os.path.exists(full_path):
            # Try alternative paths
            alt_paths = [
                os.path.join("TestMaster", module_path),
                module_path.replace("/", "\\"),
                module_path.replace("\\", "/")
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    full_path = alt_path
                    break
            else:
                print(f"Warning: Module not found: {module_path}")
                return
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics = self._calculate_function_metrics(node, full_path, content)
                    self.function_metrics.append(metrics)
                    
        except Exception as e:
            print(f"Warning: Error analyzing {full_path}: {e}")
    
    def _analyze_testmaster_functions(self):
        """Analyze functions in TestMaster directory."""
        testmaster_dir = "TestMaster"
        if not os.path.exists(testmaster_dir):
            return
            
        for root, dirs, files in os.walk(testmaster_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        tree = ast.parse(content)
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                metrics = self._calculate_function_metrics(node, file_path, content)
                                self.function_metrics.append(metrics)
                                
                    except Exception as e:
                        continue
    
    def _calculate_function_metrics(self, node: ast.FunctionDef, file_path: str, content: str) -> FunctionMetrics:
        """Calculate metrics for a function."""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
        
        function_lines = lines[start_line:end_line]
        line_count = len([line for line in function_lines if line.strip()])
        
        # Calculate complexity score (simplified)
        complexity_score = self._calculate_complexity(node)
        
        # Count parameters
        parameter_count = len(node.args.args)
        
        # Count return statements
        return_statements = len([n for n in ast.walk(node) if isinstance(n, ast.Return)])
        
        # Calculate nested depth
        nested_depth = self._calculate_nested_depth(node)
        
        # Check for docstring
        docstring_present = (isinstance(node.body[0], ast.Expr) and 
                           isinstance(node.body[0].value, ast.Constant) and 
                           isinstance(node.body[0].value.value, str)) if node.body else False
        
        # Determine modularization priority
        priority = self._determine_priority(line_count, complexity_score, nested_depth)
        
        return FunctionMetrics(
            name=node.name,
            file_path=file_path,
            line_count=line_count,
            complexity_score=complexity_score,
            parameter_count=parameter_count,
            return_statements=return_statements,
            nested_depth=nested_depth,
            docstring_present=docstring_present,
            modularization_priority=priority
        )
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_nested_depth(self, node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth."""
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
    
    def _determine_priority(self, line_count: int, complexity: int, nested_depth: int) -> str:
        """Determine modularization priority."""
        if line_count > 100 or complexity > 15 or nested_depth > 5:
            return "HIGH"
        elif line_count > 50 or complexity > 10 or nested_depth > 3:
            return "MEDIUM"
        elif line_count > 30 or complexity > 7:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_modularization_recommendations(self):
        """Generate specific modularization recommendations."""
        high_priority_functions = [fm for fm in self.function_metrics if fm.modularization_priority == "HIGH"]
        
        for func_metric in high_priority_functions:
            recommendations = self._create_function_recommendations(func_metric)
            self.recommendations.extend(recommendations)
    
    def _create_function_recommendations(self, func_metric: FunctionMetrics) -> List[ModularizationRecommendation]:
        """Create specific recommendations for a function."""
        recommendations = []
        
        if func_metric.line_count > 100:
            recommendations.append(ModularizationRecommendation(
                original_function=func_metric.name,
                file_path=func_metric.file_path,
                issue_type="OVERSIZED_FUNCTION",
                priority="HIGH",
                recommended_splits=["input_validation", "core_logic", "output_formatting"],
                new_function_names=[f"_{func_metric.name}_validate_input", 
                                  f"_{func_metric.name}_process", 
                                  f"_{func_metric.name}_format_output"],
                rationale=f"Function has {func_metric.line_count} lines, exceeds 100-line guideline"
            ))
        
        if func_metric.complexity_score > 15:
            recommendations.append(ModularizationRecommendation(
                original_function=func_metric.name,
                file_path=func_metric.file_path,
                issue_type="HIGH_COMPLEXITY",
                priority="HIGH",
                recommended_splits=["condition_checks", "main_processing", "error_handling"],
                new_function_names=[f"_{func_metric.name}_check_conditions",
                                  f"_{func_metric.name}_execute",
                                  f"_{func_metric.name}_handle_errors"],
                rationale=f"Function has complexity score {func_metric.complexity_score}, exceeds 15 threshold"
            ))
        
        if func_metric.nested_depth > 5:
            recommendations.append(ModularizationRecommendation(
                original_function=func_metric.name,
                file_path=func_metric.file_path,
                issue_type="DEEP_NESTING",
                priority="MEDIUM",
                recommended_splits=["early_returns", "extracted_conditions"],
                new_function_names=[f"_{func_metric.name}_should_continue",
                                  f"_{func_metric.name}_process_item"],
                rationale=f"Function has nesting depth {func_metric.nested_depth}, exceeds 5 levels"
            ))
        
        return recommendations
    
    def _generate_summary(self) -> Dict:
        """Generate analysis summary."""
        total_functions = len(self.function_metrics)
        high_priority = len([fm for fm in self.function_metrics if fm.modularization_priority == "HIGH"])
        medium_priority = len([fm for fm in self.function_metrics if fm.modularization_priority == "MEDIUM"])
        low_priority = len([fm for fm in self.function_metrics if fm.modularization_priority == "LOW"])
        
        avg_line_count = sum(fm.line_count for fm in self.function_metrics) / total_functions if total_functions > 0 else 0
        avg_complexity = sum(fm.complexity_score for fm in self.function_metrics) / total_functions if total_functions > 0 else 0
        
        return {
            "total_functions_analyzed": total_functions,
            "modularization_priorities": {
                "high_priority": high_priority,
                "medium_priority": medium_priority,
                "low_priority": low_priority,
                "minimal_priority": total_functions - high_priority - medium_priority - low_priority
            },
            "average_metrics": {
                "average_line_count": round(avg_line_count, 2),
                "average_complexity": round(avg_complexity, 2)
            },
            "recommendations_generated": len(self.recommendations),
            "modularization_readiness": "READY" if high_priority > 0 else "OPTIMAL"
        }

def main():
    """Main execution function."""
    analyzer = FunctionModularizationAnalyzer()
    results = analyzer.analyze_function_modularization()
    
    # Save results
    with open("TestMaster/function_modularization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Function Modularization Analysis Complete!")
    print(f"Functions analyzed: {results['analysis_metadata']['functions_analyzed']}")
    print(f"Recommendations: {results['analysis_metadata']['recommendations_generated']}")
    print(f"Results saved to: TestMaster/function_modularization_results.json")
    
    return results

if __name__ == "__main__":
    main()