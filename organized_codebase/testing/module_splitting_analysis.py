#!/usr/bin/env python3
"""
Agent B Phase 3: Hours 61-65 - Module Splitting Analysis
Identifies oversized modules and provides splitting recommendations.
"""

import ast
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class ModuleMetrics:
    """Metrics for a single module."""
    name: str
    file_path: str
    line_count: int
    function_count: int
    class_count: int
    import_count: int
    complexity_score: int
    cohesion_estimate: float
    responsibility_areas: List[str]
    splitting_priority: str

@dataclass
class ModuleSplittingRecommendation:
    """Recommendation for splitting a module."""
    original_module: str
    file_path: str
    issue_type: str
    priority: str
    recommended_splits: List[str]
    new_module_names: List[str]
    component_groups: List[str]
    rationale: str
    estimated_effort: str

class ModuleSplittingAnalyzer:
    """Analyzes modules for splitting opportunities."""
    
    def __init__(self, base_directory: str = "."):
        self.base_directory = base_directory
        self.module_metrics: List[ModuleMetrics] = []
        self.recommendations: List[ModuleSplittingRecommendation] = []
        
        # Focus on TestMaster directory and critical modules
        self.analysis_directories = [
            "TestMaster",
            "core",
            "config", 
            "api",
            "workflow"
        ]
    
    def analyze_module_splitting(self) -> Dict:
        """Main analysis method for module splitting."""
        print("Starting Module Splitting Analysis...")
        print(f"Analyzing directories: {len(self.analysis_directories)}")
        
        for directory in self.analysis_directories:
            self._analyze_directory_modules(directory)
        
        # Also analyze individual Python files in root
        self._analyze_root_modules()
        
        self._generate_splitting_recommendations()
        
        results = {
            "analysis_metadata": {
                "analyzer": "Agent B - Module Splitting Analysis",
                "phase": "Hours 61-65",
                "timestamp": datetime.now().isoformat(),
                "modules_analyzed": len(self.module_metrics),
                "recommendations_generated": len(self.recommendations)
            },
            "module_metrics": [asdict(mm) for mm in self.module_metrics],
            "splitting_recommendations": [asdict(sr) for sr in self.recommendations],
            "summary": self._generate_summary()
        }
        
        return results
    
    def _analyze_directory_modules(self, directory: str):
        """Analyze modules in a specific directory."""
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            return
            
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = os.path.join(root, file)
                    self._analyze_single_module(file_path)
    
    def _analyze_root_modules(self):
        """Analyze Python modules in root directory."""
        for file in os.listdir('.'):
            if file.endswith('.py') and not file.startswith('test_'):
                self._analyze_single_module(file)
    
    def _analyze_single_module(self, file_path: str):
        """Analyze a single module file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Skip very small files
            if len(content.strip()) < 100:
                return
            
            tree = ast.parse(content)
            metrics = self._calculate_module_metrics(tree, file_path, content)
            self.module_metrics.append(metrics)
            
        except Exception as e:
            print(f"Warning: Error analyzing {file_path}: {e}")
    
    def _calculate_module_metrics(self, tree: ast.AST, file_path: str, content: str) -> ModuleMetrics:
        """Calculate metrics for a module."""
        lines = content.split('\n')
        line_count = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # Count functions and classes
        function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        
        # Count imports
        import_count = len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
        
        # Calculate complexity score (simplified)
        complexity_score = self._calculate_module_complexity(tree)
        
        # Estimate cohesion based on naming patterns and structure
        cohesion_estimate = self._estimate_module_cohesion(tree, content)
        
        # Identify responsibility areas
        responsibility_areas = self._identify_responsibility_areas(tree, content, file_path)
        
        # Determine splitting priority
        splitting_priority = self._determine_splitting_priority(
            line_count, function_count, class_count, len(responsibility_areas)
        )
        
        module_name = Path(file_path).stem
        
        return ModuleMetrics(
            name=module_name,
            file_path=file_path,
            line_count=line_count,
            function_count=function_count,
            class_count=class_count,
            import_count=import_count,
            complexity_score=complexity_score,
            cohesion_estimate=cohesion_estimate,
            responsibility_areas=responsibility_areas,
            splitting_priority=splitting_priority
        )
    
    def _calculate_module_complexity(self, tree: ast.AST) -> int:
        """Calculate module complexity score."""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 2  # Functions add to complexity
            elif isinstance(node, ast.ClassDef):
                complexity += 3  # Classes add more complexity
        
        return complexity
    
    def _estimate_module_cohesion(self, tree: ast.AST, content: str) -> float:
        """Estimate module cohesion based on naming patterns and structure."""
        # Simple heuristic based on naming consistency
        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not function_names and not class_names:
            return 0.5
        
        # Check for common prefixes/patterns
        all_names = function_names + class_names
        if len(all_names) < 2:
            return 0.8
        
        # Count common word patterns
        common_patterns = 0
        total_comparisons = 0
        
        for i, name1 in enumerate(all_names):
            for name2 in all_names[i+1:]:
                total_comparisons += 1
                # Simple word overlap check
                words1 = set(name1.lower().split('_'))
                words2 = set(name2.lower().split('_'))
                if words1 & words2:  # Common words
                    common_patterns += 1
        
        if total_comparisons == 0:
            return 0.8
        
        cohesion_score = common_patterns / total_comparisons
        return min(0.95, max(0.3, cohesion_score))
    
    def _identify_responsibility_areas(self, tree: ast.AST, content: str, file_path: str) -> List[str]:
        """Identify different responsibility areas in the module."""
        responsibilities = set()
        
        # Analyze based on file path
        if 'test' in file_path.lower():
            responsibilities.add('testing')
        if 'config' in file_path.lower():
            responsibilities.add('configuration')
        if 'api' in file_path.lower():
            responsibilities.add('api')
        if 'data' in file_path.lower():
            responsibilities.add('data')
        if 'util' in file_path.lower():
            responsibilities.add('utilities')
        
        # Analyze function and class names
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = node.name.lower()
                if any(word in name for word in ['validate', 'check', 'verify']):
                    responsibilities.add('validation')
                if any(word in name for word in ['process', 'execute', 'run']):
                    responsibilities.add('processing')
                if any(word in name for word in ['format', 'serialize', 'parse']):
                    responsibilities.add('formatting')
                if any(word in name for word in ['load', 'save', 'read', 'write']):
                    responsibilities.add('io')
                if any(word in name for word in ['analyze', 'calculate', 'compute']):
                    responsibilities.add('analytics')
                if any(word in name for word in ['monitor', 'track', 'log']):
                    responsibilities.add('monitoring')
                
            elif isinstance(node, ast.ClassDef):
                name = node.name.lower()
                if any(word in name for word in ['manager', 'controller', 'orchestrator']):
                    responsibilities.add('coordination')
                if any(word in name for word in ['processor', 'engine', 'handler']):
                    responsibilities.add('processing')
                if any(word in name for word in ['validator', 'checker']):
                    responsibilities.add('validation')
                if any(word in name for word in ['serializer', 'formatter']):
                    responsibilities.add('formatting')
        
        # Content-based analysis for imports
        if 'import json' in content or 'import yaml' in content:
            responsibilities.add('serialization')
        if 'import logging' in content:
            responsibilities.add('logging')
        if 'import asyncio' in content:
            responsibilities.add('async_processing')
        if 'import unittest' in content or 'import pytest' in content:
            responsibilities.add('testing')
        
        return list(responsibilities)
    
    def _determine_splitting_priority(self, line_count: int, function_count: int, 
                                    class_count: int, responsibility_count: int) -> str:
        """Determine module splitting priority."""
        # High priority conditions
        if (line_count > 500 or 
            function_count > 30 or 
            class_count > 10 or 
            responsibility_count > 4):
            return "HIGH"
        
        # Medium priority conditions
        if (line_count > 300 or 
            function_count > 20 or 
            class_count > 6 or 
            responsibility_count > 3):
            return "MEDIUM"
        
        # Low priority conditions
        if (line_count > 200 or 
            function_count > 15 or 
            class_count > 4 or 
            responsibility_count > 2):
            return "LOW"
        
        return "MINIMAL"
    
    def _generate_splitting_recommendations(self):
        """Generate specific splitting recommendations."""
        high_priority_modules = [mm for mm in self.module_metrics if mm.splitting_priority == "HIGH"]
        medium_priority_modules = [mm for mm in self.module_metrics if mm.splitting_priority == "MEDIUM"]
        
        for module_metric in high_priority_modules:
            recommendations = self._create_splitting_recommendations(module_metric, "HIGH")
            self.recommendations.extend(recommendations)
            
        for module_metric in medium_priority_modules[:15]:  # Limit to top 15 medium priority
            recommendations = self._create_splitting_recommendations(module_metric, "MEDIUM")
            self.recommendations.extend(recommendations)
    
    def _create_splitting_recommendations(self, module_metric: ModuleMetrics, priority: str) -> List[ModuleSplittingRecommendation]:
        """Create specific recommendations for a module."""
        recommendations = []
        
        if module_metric.line_count > 500:
            recommendations.append(ModuleSplittingRecommendation(
                original_module=module_metric.name,
                file_path=module_metric.file_path,
                issue_type="OVERSIZED_MODULE",
                priority=priority,
                recommended_splits=["core", "utils", "config"],
                new_module_names=[f"{module_metric.name}_core", f"{module_metric.name}_utils", f"{module_metric.name}_config"],
                component_groups=["Main functionality", "Utility functions", "Configuration handling"],
                rationale=f"Module has {module_metric.line_count} lines, exceeds 500-line guideline",
                estimated_effort="3-5 days"
            ))
        
        if module_metric.function_count > 30:
            recommendations.append(ModuleSplittingRecommendation(
                original_module=module_metric.name,
                file_path=module_metric.file_path,
                issue_type="TOO_MANY_FUNCTIONS",
                priority=priority,
                recommended_splits=["operations", "helpers", "validators"],
                new_module_names=[f"{module_metric.name}_ops", f"{module_metric.name}_helpers", f"{module_metric.name}_validators"],
                component_groups=["Core operations", "Helper functions", "Validation logic"],
                rationale=f"Module has {module_metric.function_count} functions, exceeds 30 function guideline",
                estimated_effort="2-4 days"
            ))
        
        if len(module_metric.responsibility_areas) > 4:
            recommendations.append(ModuleSplittingRecommendation(
                original_module=module_metric.name,
                file_path=module_metric.file_path,
                issue_type="MULTIPLE_RESPONSIBILITIES",
                priority=priority,
                recommended_splits=module_metric.responsibility_areas[:3],  # Top 3 responsibilities
                new_module_names=[f"{module_metric.name}_{area}" for area in module_metric.responsibility_areas[:3]],
                component_groups=module_metric.responsibility_areas[:3],
                rationale=f"Module has {len(module_metric.responsibility_areas)} responsibility areas: {', '.join(module_metric.responsibility_areas)}",
                estimated_effort="4-6 days"
            ))
        
        if module_metric.cohesion_estimate < 0.5:
            recommendations.append(ModuleSplittingRecommendation(
                original_module=module_metric.name,
                file_path=module_metric.file_path,
                issue_type="LOW_COHESION",
                priority="MEDIUM" if priority == "HIGH" else "LOW",
                recommended_splits=["related_components", "separate_concerns"],
                new_module_names=[f"{module_metric.name}_core", f"{module_metric.name}_misc"],
                component_groups=["Cohesive functionality", "Separate concerns"],
                rationale=f"Module has low cohesion {module_metric.cohesion_estimate:.2f}, suggests mixed concerns",
                estimated_effort="2-3 days"
            ))
        
        return recommendations
    
    def _generate_summary(self) -> Dict:
        """Generate analysis summary."""
        total_modules = len(self.module_metrics)
        high_priority = len([mm for mm in self.module_metrics if mm.splitting_priority == "HIGH"])
        medium_priority = len([mm for mm in self.module_metrics if mm.splitting_priority == "MEDIUM"])
        low_priority = len([mm for mm in self.module_metrics if mm.splitting_priority == "LOW"])
        
        avg_line_count = sum(mm.line_count for mm in self.module_metrics) / total_modules if total_modules > 0 else 0
        avg_function_count = sum(mm.function_count for mm in self.module_metrics) / total_modules if total_modules > 0 else 0
        avg_cohesion = sum(mm.cohesion_estimate for mm in self.module_metrics) / total_modules if total_modules > 0 else 0
        
        total_responsibilities = sum(len(mm.responsibility_areas) for mm in self.module_metrics)
        avg_responsibilities = total_responsibilities / total_modules if total_modules > 0 else 0
        
        return {
            "total_modules_analyzed": total_modules,
            "splitting_priorities": {
                "high_priority": high_priority,
                "medium_priority": medium_priority,
                "low_priority": low_priority,
                "minimal_priority": total_modules - high_priority - medium_priority - low_priority
            },
            "average_metrics": {
                "average_line_count": round(avg_line_count, 2),
                "average_function_count": round(avg_function_count, 2),
                "average_cohesion": round(avg_cohesion, 2),
                "average_responsibilities": round(avg_responsibilities, 2)
            },
            "recommendations_generated": len(self.recommendations),
            "splitting_readiness": "READY" if high_priority > 0 else "OPTIMAL"
        }

def main():
    """Main execution function."""
    analyzer = ModuleSplittingAnalyzer()
    results = analyzer.analyze_module_splitting()
    
    # Save results
    with open("TestMaster/module_splitting_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Module Splitting Analysis Complete!")
    print(f"Modules analyzed: {results['analysis_metadata']['modules_analyzed']}")
    print(f"Recommendations: {results['analysis_metadata']['recommendations_generated']}")
    print(f"Results saved to: TestMaster/module_splitting_results.json")
    
    return results

if __name__ == "__main__":
    main()