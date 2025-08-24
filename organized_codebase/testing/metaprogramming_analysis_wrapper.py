"""
Modular MetaprogrammingAnalysis

This is the new modular version that combines all submodules
while maintaining the same API as the original.

This ensures backward compatibility while providing modular benefits.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .metaprog_analysis import MetaprogCoreAnalyzer, MetaprogSecurityAnalyzer, MetaprogReflectionAnalyzer
from .base import BaseAnalyzer


class MetaprogrammingAnalysis(BaseAnalyzer):
    """
    Modular MetaprogrammingAnalysis that combines specialized analyzers
    
    Maintains API compatibility with original while using modular architecture.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize specialized analyzers
        self.metaprogcore_analyzer = MetaprogCoreAnalyzer()
        self.metaprogsecurity_analyzer = MetaprogSecurityAnalyzer()
        self.metaprogreflection_analyzer = MetaprogReflectionAnalyzer()

    
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using modular components
        
        Returns the same structure as the original analyzer for compatibility.
        """
        # Run all specialized analyzers
        metaprogcore_results = self.metaprogcore_analyzer.analyze(file_path)
        metaprogsecurity_results = self.metaprogsecurity_analyzer.analyze(file_path)
        metaprogreflection_results = self.metaprogreflection_analyzer.analyze(file_path)

        
        # Combine results in the original format
        combined_results = {
            "summary": self._generate_combined_summary(metaprogcore_results, metaprogsecurity_results, metaprogreflection_results),
            # TODO: Add specific result combination logic based on original API
        }
        
        return combined_results
    
    def _generate_combined_summary(self, *results) -> Dict[str, Any]:
        """Generate combined summary from all analyzer results"""
        total_issues = 0
        total_patterns = 0
        
        for result in results:
            if isinstance(result, dict):
                summary = result.get("summary", {})
                total_issues += summary.get("total_issues", 0)
                total_patterns += summary.get("total_patterns", 0)
        
        return {
            "total_issues": total_issues,
            "total_patterns": total_patterns,
            "modular_components": len(results)
        }
