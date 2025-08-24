#!/usr/bin/env python3
"""
Agent B Phase 3: Hours 56-60 - Class Modularization Analysis
Identifies oversized classes and provides modularization recommendations.
"""

import ast
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    name: str
    file_path: str
    line_count: int
    method_count: int
    attribute_count: int
    inheritance_depth: int
    responsibility_score: int
    cohesion_estimate: float
    docstring_present: bool
    modularization_priority: str

@dataclass
class ClassModularizationRecommendation:
    """Recommendation for modularizing a class."""
    original_class: str
    file_path: str
    issue_type: str
    priority: str
    recommended_splits: List[str]
    new_class_names: List[str]
    extracted_responsibilities: List[str]
    rationale: str

class ClassModularizationAnalyzer:
    """Analyzes classes for modularization opportunities."""
    
    def __init__(self, base_directory: str = "."):
        self.base_directory = base_directory
        self.class_metrics: List[ClassMetrics] = []
        self.recommendations: List[ClassModularizationRecommendation] = []
        
        # Critical modules identified in Phase 2
        self.critical_modules = [
            "core/intelligence/__init__.py",
            "core/intelligence/testing/__init__.py", 
            "core/intelligence/analytics/__init__.py",
            "testmaster_orchestrator.py",
            "agentic_test_monitor.py",
            "config/__init__.py"
        ]
    
    def analyze_class_modularization(self) -> Dict:
        """Main analysis method for class modularization."""
        print("Starting Class Modularization Analysis...")
        print(f"Analyzing critical modules: {len(self.critical_modules)}")
        
        for module_path in self.critical_modules:
            self._analyze_module_classes(module_path)
        
        # Also analyze Python files in TestMaster directory
        self._analyze_testmaster_classes()
        
        self._generate_modularization_recommendations()
        
        results = {
            "analysis_metadata": {
                "analyzer": "Agent B - Class Modularization Analysis",
                "phase": "Hours 56-60",
                "timestamp": datetime.now().isoformat(),
                "classes_analyzed": len(self.class_metrics),
                "recommendations_generated": len(self.recommendations)
            },
            "class_metrics": [asdict(cm) for cm in self.class_metrics],
            "modularization_recommendations": [asdict(mr) for mr in self.recommendations],
            "summary": self._generate_summary()
        }
        
        return results
    
    def _analyze_module_classes(self, module_path: str):
        """Analyze classes in a specific module."""
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
                if isinstance(node, ast.ClassDef):
                    metrics = self._calculate_class_metrics(node, full_path, content)
                    self.class_metrics.append(metrics)
                    
        except Exception as e:
            print(f"Warning: Error analyzing {full_path}: {e}")
    
    def _analyze_testmaster_classes(self):
        """Analyze classes in TestMaster directory."""
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
                            if isinstance(node, ast.ClassDef):
                                metrics = self._calculate_class_metrics(node, file_path, content)
                                self.class_metrics.append(metrics)
                                
                    except Exception as e:
                        continue
    
    def _calculate_class_metrics(self, node: ast.ClassDef, file_path: str, content: str) -> ClassMetrics:
        """Calculate metrics for a class."""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 50
        
        class_lines = lines[start_line:end_line]
        line_count = len([line for line in class_lines if line.strip() and not line.strip().startswith('#')])
        
        # Count methods
        method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
        
        # Count attributes (simplified - instance variables in __init__)
        attribute_count = self._count_class_attributes(node)
        
        # Calculate inheritance depth
        inheritance_depth = len(node.bases)
        
        # Calculate responsibility score (methods + attributes as proxy)
        responsibility_score = method_count + attribute_count
        
        # Estimate cohesion (simplified - based on method/attribute ratio)
        cohesion_estimate = self._estimate_cohesion(node)
        
        # Check for docstring
        docstring_present = (isinstance(node.body[0], ast.Expr) and 
                           isinstance(node.body[0].value, ast.Constant) and 
                           isinstance(node.body[0].value.value, str)) if node.body else False
        
        # Determine modularization priority
        priority = self._determine_class_priority(line_count, method_count, responsibility_score)
        
        return ClassMetrics(
            name=node.name,
            file_path=file_path,
            line_count=line_count,
            method_count=method_count,
            attribute_count=attribute_count,
            inheritance_depth=inheritance_depth,
            responsibility_score=responsibility_score,
            cohesion_estimate=cohesion_estimate,
            docstring_present=docstring_present,
            modularization_priority=priority
        )
    
    def _count_class_attributes(self, node: ast.ClassDef) -> int:
        """Count class attributes (simplified analysis)."""
        attribute_count = 0
        
        # Look for __init__ method and count self.attribute assignments
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                                attribute_count += 1
        
        # Also count class-level assignments
        for item in node.body:
            if isinstance(item, ast.Assign):
                attribute_count += len(item.targets)
        
        return attribute_count
    
    def _estimate_cohesion(self, node: ast.ClassDef) -> float:
        """Estimate class cohesion (simplified)."""
        method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
        attribute_count = self._count_class_attributes(node)
        
        if method_count == 0:
            return 0.0
        
        # Simple heuristic: balanced method/attribute ratio indicates better cohesion
        if attribute_count == 0:
            return 0.5  # Methods only, moderate cohesion
        
        ratio = method_count / attribute_count
        if 0.5 <= ratio <= 3.0:  # Good balance
            return 0.8
        elif ratio < 0.5:  # Too many attributes
            return 0.6
        else:  # Too many methods relative to state
            return 0.7
    
    def _determine_class_priority(self, line_count: int, method_count: int, responsibility_score: int) -> str:
        """Determine class modularization priority."""
        if line_count > 300 or method_count > 20 or responsibility_score > 25:
            return "HIGH"
        elif line_count > 200 or method_count > 15 or responsibility_score > 18:
            return "MEDIUM"
        elif line_count > 100 or method_count > 10 or responsibility_score > 12:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_modularization_recommendations(self):
        """Generate specific modularization recommendations."""
        high_priority_classes = [cm for cm in self.class_metrics if cm.modularization_priority == "HIGH"]
        medium_priority_classes = [cm for cm in self.class_metrics if cm.modularization_priority == "MEDIUM"]
        
        for class_metric in high_priority_classes:
            recommendations = self._create_class_recommendations(class_metric, "HIGH")
            self.recommendations.extend(recommendations)
            
        for class_metric in medium_priority_classes[:10]:  # Limit to top 10 medium priority
            recommendations = self._create_class_recommendations(class_metric, "MEDIUM")
            self.recommendations.extend(recommendations)
    
    def _create_class_recommendations(self, class_metric: ClassMetrics, priority: str) -> List[ClassModularizationRecommendation]:
        """Create specific recommendations for a class."""
        recommendations = []
        
        if class_metric.line_count > 300:
            recommendations.append(ClassModularizationRecommendation(
                original_class=class_metric.name,
                file_path=class_metric.file_path,
                issue_type="OVERSIZED_CLASS",
                priority=priority,
                recommended_splits=["core_functionality", "data_management", "interface_methods"],
                new_class_names=[f"{class_metric.name}Core", f"{class_metric.name}Data", f"{class_metric.name}Interface"],
                extracted_responsibilities=["Business logic", "Data operations", "External interfaces"],
                rationale=f"Class has {class_metric.line_count} lines, exceeds 300-line guideline"
            ))
        
        if class_metric.method_count > 20:
            recommendations.append(ClassModularizationRecommendation(
                original_class=class_metric.name,
                file_path=class_metric.file_path,
                issue_type="TOO_MANY_METHODS",
                priority=priority,
                recommended_splits=["primary_methods", "utility_methods", "validation_methods"],
                new_class_names=[f"{class_metric.name}Primary", f"{class_metric.name}Utils", f"{class_metric.name}Validator"],
                extracted_responsibilities=["Core operations", "Helper functions", "Input validation"],
                rationale=f"Class has {class_metric.method_count} methods, exceeds 20 method guideline"
            ))
        
        if class_metric.responsibility_score > 25:
            recommendations.append(ClassModularizationRecommendation(
                original_class=class_metric.name,
                file_path=class_metric.file_path,
                issue_type="MULTIPLE_RESPONSIBILITIES",
                priority=priority,
                recommended_splits=["domain_logic", "infrastructure", "coordination"],
                new_class_names=[f"{class_metric.name}Domain", f"{class_metric.name}Infrastructure", f"{class_metric.name}Coordinator"],
                extracted_responsibilities=["Business domain logic", "Infrastructure concerns", "Component coordination"],
                rationale=f"Class has responsibility score {class_metric.responsibility_score}, indicates multiple responsibilities"
            ))
        
        if class_metric.cohesion_estimate < 0.6:
            recommendations.append(ClassModularizationRecommendation(
                original_class=class_metric.name,
                file_path=class_metric.file_path,
                issue_type="LOW_COHESION",
                priority="MEDIUM" if priority == "HIGH" else "LOW",
                recommended_splits=["related_functionality", "separate_concerns"],
                new_class_names=[f"{class_metric.name}Core", f"{class_metric.name}Support"],
                extracted_responsibilities=["Tightly related operations", "Supporting functionality"],
                rationale=f"Class has low cohesion estimate {class_metric.cohesion_estimate:.2f}, suggests mixed responsibilities"
            ))
        
        return recommendations
    
    def _generate_summary(self) -> Dict:
        """Generate analysis summary."""
        total_classes = len(self.class_metrics)
        high_priority = len([cm for cm in self.class_metrics if cm.modularization_priority == "HIGH"])
        medium_priority = len([cm for cm in self.class_metrics if cm.modularization_priority == "MEDIUM"])
        low_priority = len([cm for cm in self.class_metrics if cm.modularization_priority == "LOW"])
        
        avg_line_count = sum(cm.line_count for cm in self.class_metrics) / total_classes if total_classes > 0 else 0
        avg_method_count = sum(cm.method_count for cm in self.class_metrics) / total_classes if total_classes > 0 else 0
        avg_cohesion = sum(cm.cohesion_estimate for cm in self.class_metrics) / total_classes if total_classes > 0 else 0
        
        return {
            "total_classes_analyzed": total_classes,
            "modularization_priorities": {
                "high_priority": high_priority,
                "medium_priority": medium_priority,
                "low_priority": low_priority,
                "minimal_priority": total_classes - high_priority - medium_priority - low_priority
            },
            "average_metrics": {
                "average_line_count": round(avg_line_count, 2),
                "average_method_count": round(avg_method_count, 2),
                "average_cohesion": round(avg_cohesion, 2)
            },
            "recommendations_generated": len(self.recommendations),
            "modularization_readiness": "READY" if high_priority > 0 else "OPTIMAL"
        }

def main():
    """Main execution function."""
    analyzer = ClassModularizationAnalyzer()
    results = analyzer.analyze_class_modularization()
    
    # Save results
    with open("TestMaster/class_modularization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Class Modularization Analysis Complete!")
    print(f"Classes analyzed: {results['analysis_metadata']['classes_analyzed']}")
    print(f"Recommendations: {results['analysis_metadata']['recommendations_generated']}")
    print(f"Results saved to: TestMaster/class_modularization_results.json")
    
    return results

if __name__ == "__main__":
    main()