#!/usr/bin/env python3
"""
Agent B Phase 3: Hours 66-70 - Documentation Enhancement Analysis
Enhances all documentation with practical examples, performance guidance, and troubleshooting.
"""

import ast
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class DocumentationMetrics:
    """Metrics for documentation coverage and quality."""
    module_name: str
    file_path: str
    total_functions: int
    documented_functions: int
    total_classes: int
    documented_classes: int
    has_module_docstring: bool
    has_examples: bool
    has_performance_notes: bool
    has_troubleshooting: bool
    documentation_score: float
    enhancement_priority: str

@dataclass
class DocumentationEnhancement:
    """Enhancement recommendation for documentation."""
    target_file: str
    enhancement_type: str
    priority: str
    missing_elements: List[str]
    recommended_additions: List[str]
    example_templates: List[str]
    performance_guidance: List[str]
    troubleshooting_scenarios: List[str]

class DocumentationEnhancementAnalyzer:
    """Analyzes and enhances framework documentation."""
    
    def __init__(self, base_directory: str = "."):
        self.base_directory = base_directory
        self.documentation_metrics: List[DocumentationMetrics] = []
        self.enhancements: List[DocumentationEnhancement] = []
        
        # Focus on critical modules and TestMaster directory
        self.analysis_directories = [
            "TestMaster",
            "core/intelligence",
            "config",
            "api"
        ]
        
        # Critical modules from modularization analysis
        self.priority_modules = [
            "intelligence_orchestrator",
            "ml_infrastructure_manager", 
            "analytics_processing_engine",
            "workflow_coordination_system",
            "data_management_hub"
        ]
    
    def analyze_documentation_enhancement(self) -> Dict:
        """Main analysis method for documentation enhancement."""
        print("Starting Documentation Enhancement Analysis...")
        print(f"Analyzing directories: {len(self.analysis_directories)}")
        print(f"Priority modules: {len(self.priority_modules)}")
        
        for directory in self.analysis_directories:
            self._analyze_directory_documentation(directory)
        
        # Also analyze root Python files
        self._analyze_root_documentation()
        
        self._generate_enhancement_recommendations()
        
        results = {
            "analysis_metadata": {
                "analyzer": "Agent B - Documentation Enhancement Analysis",
                "phase": "Hours 66-70",
                "timestamp": datetime.now().isoformat(),
                "files_analyzed": len(self.documentation_metrics),
                "enhancements_generated": len(self.enhancements)
            },
            "documentation_metrics": [asdict(dm) for dm in self.documentation_metrics],
            "enhancement_recommendations": [asdict(de) for de in self.enhancements],
            "summary": self._generate_summary()
        }
        
        return results
    
    def _analyze_directory_documentation(self, directory: str):
        """Analyze documentation in a specific directory."""
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            return
            
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = os.path.join(root, file)
                    self._analyze_file_documentation(file_path)
    
    def _analyze_root_documentation(self):
        """Analyze documentation in root Python files."""
        for file in os.listdir('.'):
            if file.endswith('.py') and not file.startswith('test_'):
                self._analyze_file_documentation(file)
    
    def _analyze_file_documentation(self, file_path: str):
        """Analyze documentation in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Skip very small files
            if len(content.strip()) < 100:
                return
            
            tree = ast.parse(content)
            metrics = self._calculate_documentation_metrics(tree, file_path, content)
            self.documentation_metrics.append(metrics)
            
        except Exception as e:
            print(f"Warning: Error analyzing {file_path}: {e}")
    
    def _calculate_documentation_metrics(self, tree: ast.AST, file_path: str, content: str) -> DocumentationMetrics:
        """Calculate documentation metrics for a file."""
        # Count functions and their documentation
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        total_functions = len(functions)
        documented_functions = sum(1 for func in functions if self._has_docstring(func))
        
        # Count classes and their documentation
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        total_classes = len(classes)
        documented_classes = sum(1 for cls in classes if self._has_docstring(cls))
        
        # Check for module-level docstring
        has_module_docstring = self._has_module_docstring(tree)
        
        # Check for examples in content
        has_examples = self._has_examples(content)
        
        # Check for performance notes
        has_performance_notes = self._has_performance_notes(content)
        
        # Check for troubleshooting guidance
        has_troubleshooting = self._has_troubleshooting(content)
        
        # Calculate documentation score
        documentation_score = self._calculate_documentation_score(
            total_functions, documented_functions, total_classes, documented_classes,
            has_module_docstring, has_examples, has_performance_notes, has_troubleshooting
        )
        
        # Determine enhancement priority
        enhancement_priority = self._determine_enhancement_priority(
            documentation_score, file_path, total_functions + total_classes
        )
        
        module_name = Path(file_path).stem
        
        return DocumentationMetrics(
            module_name=module_name,
            file_path=file_path,
            total_functions=total_functions,
            documented_functions=documented_functions,
            total_classes=total_classes,
            documented_classes=documented_classes,
            has_module_docstring=has_module_docstring,
            has_examples=has_examples,
            has_performance_notes=has_performance_notes,
            has_troubleshooting=has_troubleshooting,
            documentation_score=documentation_score,
            enhancement_priority=enhancement_priority
        )
    
    def _has_docstring(self, node) -> bool:
        """Check if a function or class has a docstring."""
        if not node.body:
            return False
        first_stmt = node.body[0]
        return (isinstance(first_stmt, ast.Expr) and 
                isinstance(first_stmt.value, ast.Constant) and 
                isinstance(first_stmt.value.value, str))
    
    def _has_module_docstring(self, tree: ast.AST) -> bool:
        """Check if module has a docstring."""
        if not tree.body:
            return False
        first_stmt = tree.body[0]
        return (isinstance(first_stmt, ast.Expr) and 
                isinstance(first_stmt.value, ast.Constant) and 
                isinstance(first_stmt.value.value, str))
    
    def _has_examples(self, content: str) -> bool:
        """Check if content has usage examples."""
        example_indicators = [
            ">>> ", "Example:", "Usage:", "example:", "usage:",
            "```python", "def example", "# Example", "# Usage"
        ]
        return any(indicator in content for indicator in example_indicators)
    
    def _has_performance_notes(self, content: str) -> bool:
        """Check if content has performance guidance."""
        performance_indicators = [
            "performance", "Performance", "optimization", "Optimization",
            "efficiency", "Efficiency", "speed", "Speed", "memory", "Memory",
            "O(", "complexity", "Complexity", "benchmark", "Benchmark"
        ]
        return any(indicator in content for indicator in performance_indicators)
    
    def _has_troubleshooting(self, content: str) -> bool:
        """Check if content has troubleshooting guidance."""
        troubleshooting_indicators = [
            "troubleshoot", "Troubleshoot", "debug", "Debug", "error", "Error",
            "issue", "Issue", "problem", "Problem", "fix", "Fix", "solution", "Solution",
            "Warning:", "Note:", "Important:", "Caution:"
        ]
        return any(indicator in content for indicator in troubleshooting_indicators)
    
    def _calculate_documentation_score(self, total_functions: int, documented_functions: int,
                                     total_classes: int, documented_classes: int,
                                     has_module_docstring: bool, has_examples: bool,
                                     has_performance_notes: bool, has_troubleshooting: bool) -> float:
        """Calculate overall documentation score (0-100)."""
        score = 0.0
        
        # Function documentation (30 points)
        if total_functions > 0:
            score += (documented_functions / total_functions) * 30
        else:
            score += 30  # No functions to document
        
        # Class documentation (25 points)
        if total_classes > 0:
            score += (documented_classes / total_classes) * 25
        else:
            score += 25  # No classes to document
        
        # Module docstring (15 points)
        if has_module_docstring:
            score += 15
        
        # Examples (15 points)
        if has_examples:
            score += 15
        
        # Performance notes (10 points)
        if has_performance_notes:
            score += 10
        
        # Troubleshooting (5 points)
        if has_troubleshooting:
            score += 5
        
        return min(100.0, score)
    
    def _determine_enhancement_priority(self, documentation_score: float, 
                                      file_path: str, component_count: int) -> str:
        """Determine documentation enhancement priority."""
        # Check if it's a priority module
        is_priority = any(priority_mod in file_path.lower() for priority_mod in self.priority_modules)
        
        if documentation_score < 50 or (is_priority and documentation_score < 75):
            return "HIGH"
        elif documentation_score < 70 or (component_count > 10 and documentation_score < 80):
            return "MEDIUM"
        elif documentation_score < 85:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_enhancement_recommendations(self):
        """Generate specific enhancement recommendations."""
        high_priority = [dm for dm in self.documentation_metrics if dm.enhancement_priority == "HIGH"]
        medium_priority = [dm for dm in self.documentation_metrics if dm.enhancement_priority == "MEDIUM"]
        
        for metric in high_priority:
            enhancement = self._create_enhancement_recommendation(metric, "HIGH")
            self.enhancements.append(enhancement)
            
        for metric in medium_priority[:20]:  # Limit to top 20 medium priority
            enhancement = self._create_enhancement_recommendation(metric, "MEDIUM")
            self.enhancements.append(enhancement)
    
    def _create_enhancement_recommendation(self, metric: DocumentationMetrics, priority: str) -> DocumentationEnhancement:
        """Create specific enhancement recommendation."""
        missing_elements = []
        recommended_additions = []
        example_templates = []
        performance_guidance = []
        troubleshooting_scenarios = []
        
        # Check what's missing and recommend additions
        if not metric.has_module_docstring:
            missing_elements.append("Module docstring")
            recommended_additions.append("Comprehensive module overview with purpose and usage")
        
        if metric.documented_functions < metric.total_functions:
            missing_elements.append(f"{metric.total_functions - metric.documented_functions} function docstrings")
            recommended_additions.append("Complete function documentation with parameters and return values")
        
        if metric.documented_classes < metric.total_classes:
            missing_elements.append(f"{metric.total_classes - metric.documented_classes} class docstrings")
            recommended_additions.append("Class documentation with attributes and methods overview")
        
        if not metric.has_examples:
            missing_elements.append("Usage examples")
            example_templates.extend([
                "Basic usage example with common parameters",
                "Advanced usage example with complex scenarios",
                "Integration example with other framework components"
            ])
        
        if not metric.has_performance_notes:
            missing_elements.append("Performance guidance")
            performance_guidance.extend([
                "Performance characteristics and complexity analysis",
                "Memory usage patterns and optimization tips",
                "Benchmarking results and comparison metrics"
            ])
        
        if not metric.has_troubleshooting:
            missing_elements.append("Troubleshooting guidance")
            troubleshooting_scenarios.extend([
                "Common error scenarios and solutions",
                "Debugging tips and diagnostic techniques",
                "Configuration issues and resolution steps"
            ])
        
        # Determine enhancement type
        if metric.documentation_score < 30:
            enhancement_type = "COMPREHENSIVE_OVERHAUL"
        elif metric.documentation_score < 60:
            enhancement_type = "MAJOR_ENHANCEMENT"
        else:
            enhancement_type = "TARGETED_IMPROVEMENT"
        
        return DocumentationEnhancement(
            target_file=metric.file_path,
            enhancement_type=enhancement_type,
            priority=priority,
            missing_elements=missing_elements,
            recommended_additions=recommended_additions,
            example_templates=example_templates,
            performance_guidance=performance_guidance,
            troubleshooting_scenarios=troubleshooting_scenarios
        )
    
    def _generate_summary(self) -> Dict:
        """Generate analysis summary."""
        total_files = len(self.documentation_metrics)
        high_priority = len([dm for dm in self.documentation_metrics if dm.enhancement_priority == "HIGH"])
        medium_priority = len([dm for dm in self.documentation_metrics if dm.enhancement_priority == "MEDIUM"])
        low_priority = len([dm for dm in self.documentation_metrics if dm.enhancement_priority == "LOW"])
        
        avg_score = sum(dm.documentation_score for dm in self.documentation_metrics) / total_files if total_files > 0 else 0
        
        total_functions = sum(dm.total_functions for dm in self.documentation_metrics)
        documented_functions = sum(dm.documented_functions for dm in self.documentation_metrics)
        function_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 100
        
        total_classes = sum(dm.total_classes for dm in self.documentation_metrics)
        documented_classes = sum(dm.documented_classes for dm in self.documentation_metrics)
        class_coverage = (documented_classes / total_classes * 100) if total_classes > 0 else 100
        
        files_with_examples = len([dm for dm in self.documentation_metrics if dm.has_examples])
        files_with_performance = len([dm for dm in self.documentation_metrics if dm.has_performance_notes])
        files_with_troubleshooting = len([dm for dm in self.documentation_metrics if dm.has_troubleshooting])
        
        return {
            "total_files_analyzed": total_files,
            "enhancement_priorities": {
                "high_priority": high_priority,
                "medium_priority": medium_priority,
                "low_priority": low_priority,
                "minimal_priority": total_files - high_priority - medium_priority - low_priority
            },
            "coverage_metrics": {
                "average_documentation_score": round(avg_score, 2),
                "function_documentation_coverage": round(function_coverage, 2),
                "class_documentation_coverage": round(class_coverage, 2),
                "files_with_examples_percentage": round((files_with_examples / total_files * 100), 2),
                "files_with_performance_notes_percentage": round((files_with_performance / total_files * 100), 2),
                "files_with_troubleshooting_percentage": round((files_with_troubleshooting / total_files * 100), 2)
            },
            "enhancements_generated": len(self.enhancements),
            "documentation_readiness": "NEEDS_ENHANCEMENT" if high_priority > 0 else "GOOD"
        }

def main():
    """Main execution function."""
    analyzer = DocumentationEnhancementAnalyzer()
    results = analyzer.analyze_documentation_enhancement()
    
    # Save results
    with open("TestMaster/documentation_enhancement_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Documentation Enhancement Analysis Complete!")
    print(f"Files analyzed: {results['analysis_metadata']['files_analyzed']}")
    print(f"Enhancements: {results['analysis_metadata']['enhancements_generated']}")
    print(f"Average score: {results['summary']['coverage_metrics']['average_documentation_score']:.1f}/100")
    print(f"Results saved to: TestMaster/documentation_enhancement_results.json")
    
    return results

if __name__ == "__main__":
    main()