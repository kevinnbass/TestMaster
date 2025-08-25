"""
Modular ML Code Analysis

This is the new modular version that combines all ML analysis submodules
while maintaining the same API as the original ml_code_analysis.py

This ensures backward compatibility while providing modular benefits.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .ml_analysis import MLCoreAnalyzer, MLTensorAnalyzer, MLModelAnalyzer, MLDataAnalyzer
from .base import BaseAnalyzer


class MLCodeAnalyzer(BaseAnalyzer):
    """
    Modular ML Code Analyzer that combines specialized analyzers
    
    Maintains API compatibility with original ml_code_analysis.py while
    using the new modular architecture internally.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize specialized analyzers
        self.core_analyzer = MLCoreAnalyzer()
        self.tensor_analyzer = MLTensorAnalyzer()
        self.model_analyzer = MLModelAnalyzer()
        self.data_analyzer = MLDataAnalyzer()
    
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive ML code analysis using modular components
        
        Returns the same structure as the original analyzer for compatibility.
        """
        # Run all specialized analyzers
        core_results = self.core_analyzer.analyze(file_path)
        tensor_results = self.tensor_analyzer.analyze(file_path)
        model_results = self.model_analyzer.analyze(file_path)
        data_results = self.data_analyzer.analyze(file_path)
        
        # Combine results in the original format
        combined_results = {
            # Framework detection from core analyzer
            "framework_detection": core_results.get("framework_detection", {}),
            
            # Tensor analysis
            "tensor_shape_analysis": {
                "issues": tensor_results.get("tensor_issues", []),
                "operations": tensor_results.get("tensor_operations", {}),
                "shape_analysis": tensor_results.get("shape_analysis", {}),
                "memory_analysis": tensor_results.get("memory_analysis", {})
            },
            
            # Model architecture from model analyzer
            "model_architecture": {
                "architectures": model_results.get("model_architectures", {}),
                "training_patterns": model_results.get("training_patterns", {}),
                "hyperparameter_analysis": model_results.get("hyperparameter_analysis", {})
            },
            
            # Data pipeline from data analyzer
            "data_pipeline": {
                "patterns": data_results.get("data_patterns", {}),
                "preprocessing": data_results.get("preprocessing_analysis", {}),
                "leakage_analysis": data_results.get("data_leakage_analysis", {})
            },
            
            # Training loop analysis from model analyzer
            "training_loop_analysis": model_results.get("training_patterns", {}),
            
            # Data leakage detection from data analyzer
            "data_leakage_detection": data_results.get("data_leakage_analysis", {}),
            
            # Preprocessing analysis from data analyzer
            "preprocessing_analysis": data_results.get("preprocessing_analysis", {}),
            
            # Hyperparameter analysis from model analyzer
            "hyperparameter_analysis": model_results.get("hyperparameter_analysis", {}),
            
            # GPU optimization from tensor analyzer
            "gpu_optimization": {
                "gpu_issues": [i for i in tensor_results.get("tensor_issues", []) 
                              if "gpu" in i.get("type", "").lower()],
                "optimization_opportunities": tensor_results.get("recommendations", [])
            },
            
            # Model serialization from model analyzer
            "model_serialization": {
                "patterns": [p for p in model_results.get("training_patterns", {}).get("all_patterns", [])
                           if p.get("type") == "serialization"]
            },
            
            # Reproducibility check from core analyzer
            "reproducibility_check": self._extract_reproducibility_info(core_results),
            
            # Performance bottlenecks from all analyzers
            "performance_bottlenecks": self._combine_performance_issues(
                core_results, tensor_results, model_results, data_results
            ),
            
            # Best practices from all analyzers
            "best_practices": self._combine_best_practices(
                core_results, tensor_results, model_results, data_results
            ),
            
            # Security analysis from core analyzer
            "security_analysis": self._extract_security_info(core_results),
            
            # Summary combining all results
            "summary": self._generate_combined_summary(
                core_results, tensor_results, model_results, data_results
            )
        }
        
        return combined_results
    
    def _extract_reproducibility_info(self, core_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reproducibility information from core results"""
        reproducibility_issues = [
            issue for issue in core_results.get("issues", [])
            if issue.get("type") == "reproducibility"
        ]
        
        return {
            "issues": reproducibility_issues,
            "has_reproducibility_issues": len(reproducibility_issues) > 0,
            "recommendations": [
                "Set random seeds for reproducible results",
                "Pin dependency versions",
                "Use deterministic algorithms where possible"
            ]
        }
    
    def _extract_security_info(self, core_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract security information from core results"""
        security_issues = [
            issue for issue in core_results.get("issues", [])
            if "security" in issue.get("impact", "").lower()
        ]
        
        return {
            "issues": security_issues,
            "has_security_issues": len(security_issues) > 0,
            "recommendations": [
                "Validate model inputs",
                "Secure model artifacts",
                "Use secure data loading practices"
            ]
        }
    
    def _combine_performance_issues(self, *results) -> Dict[str, Any]:
        """Combine performance issues from all analyzers"""
        all_performance_issues = []
        
        for result in results:
            if isinstance(result, dict):
                issues = result.get("issues", [])
                if isinstance(issues, list):
                    performance_issues = [
                        issue for issue in issues
                        if issue.get("impact") == "performance" or 
                           "performance" in issue.get("type", "").lower()
                    ]
                    all_performance_issues.extend(performance_issues)
        
        return {
            "total_issues": len(all_performance_issues),
            "issues": all_performance_issues,
            "categories": list(set(issue.get("type", "") for issue in all_performance_issues))
        }
    
    def _combine_best_practices(self, *results) -> Dict[str, Any]:
        """Combine best practice recommendations from all analyzers"""
        all_recommendations = []
        
        for result in results:
            if isinstance(result, dict):
                recommendations = result.get("recommendations", [])
                if isinstance(recommendations, list):
                    all_recommendations.extend(recommendations)
        
        # Deduplicate recommendations
        unique_recommendations = []
        seen_recommendations = set()
        
        for rec in all_recommendations:
            if isinstance(rec, dict):
                rec_text = rec.get("recommendation", "")
                if rec_text and rec_text not in seen_recommendations:
                    unique_recommendations.append(rec)
                    seen_recommendations.add(rec_text)
        
        return {
            "total_recommendations": len(unique_recommendations),
            "recommendations": unique_recommendations,
            "categories": list(set(rec.get("category", "") for rec in unique_recommendations))
        }
    
    def _generate_combined_summary(self, *results) -> Dict[str, Any]:
        """Generate combined summary from all analyzer results"""
        total_issues = 0
        total_patterns = 0
        total_files = set()
        
        frameworks_detected = set()
        issue_severities = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for result in results:
            if isinstance(result, dict):
                summary = result.get("summary", {})
                
                # Count issues
                if "total_issues" in summary:
                    total_issues += summary["total_issues"]
                elif "total_tensor_issues" in summary:
                    total_issues += summary["total_tensor_issues"]
                elif "total_model_issues" in summary:
                    total_issues += summary["total_model_issues"]
                elif "total_data_issues" in summary:
                    total_issues += summary["total_data_issues"]
                
                # Count patterns
                for key in summary:
                    if "patterns" in key or "operations" in key:
                        total_patterns += summary.get(key, 0)
                
                # Count files
                if "files_analyzed" in summary:
                    total_files.add(summary["files_analyzed"])
                
                # Collect frameworks
                if "frameworks_detected" in summary:
                    frameworks_detected.update(summary["frameworks_detected"])
                
                # Count severities
                for severity in issue_severities:
                    if f"{severity}_issues" in summary:
                        issue_severities[severity] += summary[f"{severity}_issues"]
        
        return {
            "total_issues": total_issues,
            "total_patterns": total_patterns,
            "total_files": len(total_files),
            "frameworks_detected": list(frameworks_detected),
            "issue_severity_distribution": issue_severities,
            "analysis_coverage": {
                "framework_detection": True,
                "tensor_analysis": True,
                "model_analysis": True,
                "data_analysis": True
            }
        }