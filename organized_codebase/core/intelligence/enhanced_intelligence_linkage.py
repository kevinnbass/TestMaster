#!/usr/bin/env python3
"""
Enhanced Multi-Dimensional Functional Linkage Analysis
=======================================================

Integrates the comprehensive intelligence suite to provide rich, multi-layered
analysis beyond simple import counting, including semantic analysis, security 
assessment, quality metrics, and pattern recognition.

Author: Claude Code
"""

import os
import sys
import json
import ast
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
from datetime import timedelta

# STEELCLAD Phase 2: Import extracted analyzer modules
from .analysis.semantic_analyzer import create_semantic_analyzer
from .analysis.security_analyzer import create_security_analyzer
from .analysis.quality_analyzer import create_quality_analyzer
from .analysis.pattern_analyzer import create_pattern_analyzer
from .analysis.predictive_analyzer import create_predictive_analyzer, PredictiveMetric, ConfidenceLevel

# Add TestMaster to Python path
testmaster_dir = Path(__file__).parent / "TestMaster"
sys.path.insert(0, str(testmaster_dir))

# IRONCLAD CONSOLIDATION: Predictive Analytics Integration
class PredictionType(Enum):
    """Types of predictions available"""
    HEALTH_TREND = "health_trend"
    SERVICE_FAILURE = "service_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_UTILIZATION = "resource_utilization"
    DEPENDENCY_ISSUES = "dependency_issues"

class EnhancedLinkageAnalyzer:
    """Multi-dimensional linkage analyzer using the full intelligence suite."""
    
    def __init__(self):
        # STEELCLAD Phase 2: Initialize extracted analyzer modules
        self.semantic_analyzer = create_semantic_analyzer()
        self.security_analyzer = create_security_analyzer()
        self.quality_analyzer = create_quality_analyzer()
        self.pattern_analyzer = create_pattern_analyzer()
        self.predictive_analyzer = create_predictive_analyzer()
        
        # Legacy patterns for compatibility
        self.semantic_patterns = self._load_semantic_patterns()
        self.security_patterns = self._load_security_patterns()
        self.quality_thresholds = self._load_quality_thresholds()
        
    def analyze_codebase(self, base_dir="TestMaster", max_files=1000):
        """Comprehensive multi-dimensional linkage analysis."""
        
        results = {
            "basic_linkage": self._basic_linkage_analysis(base_dir, max_files),
            "semantic_dimensions": {},
            "security_dimensions": {},
            "quality_dimensions": {},
            "pattern_dimensions": {},
            "predictive_dimensions": {},
            "multi_layer_graph": {
                "nodes": [],
                "links": [],
                "layers": []
            },
            "intelligence_summary": {},
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Get file list from basic analysis
        python_files = self._get_python_files(base_dir, max_files)
        
        print(f"Agent Alpha Enhanced Analysis: Processing {len(python_files)} files...")
        
        # Analyze each dimension with progress tracking
        print("Phase 1/6: Semantic Analysis...")
        results["semantic_dimensions"] = self._semantic_analysis(python_files, base_dir)
        print("Phase 2/6: Security Analysis...")
        results["security_dimensions"] = self._security_analysis(python_files, base_dir)  
        print("Phase 3/6: Quality Analysis...")
        results["quality_dimensions"] = self._quality_analysis(python_files, base_dir)
        print("Phase 4/6: Pattern Analysis...")
        results["pattern_dimensions"] = self._pattern_analysis(python_files, base_dir)
        print("Phase 5/6: Predictive Analysis...")
        results["predictive_dimensions"] = self._predictive_analysis(python_files, base_dir)
        
        # Build multi-layer graph
        print("Phase 6/6: Building Multi-layer Graph...")
        results["multi_layer_graph"] = self._build_multi_layer_graph(python_files, base_dir, results)
        
        # Generate intelligence summary
        print("Finalizing Intelligence Summary...")
        results["intelligence_summary"] = self._generate_intelligence_summary(results)
        
        print(f"Agent Alpha Analysis Complete: {len(python_files)} files processed!")
        
        return results
    
    def _basic_linkage_analysis(self, base_dir, max_files):
        """Original linkage analysis for comparison."""
        from enhanced_linkage_dashboard import quick_linkage_analysis
        return quick_linkage_analysis(base_dir, max_files)
    
    def _get_python_files(self, base_dir, max_files):
        """Get list of Python files to analyze."""
        python_files = []
        base_path = Path(base_dir)
        
        if not base_path.exists():
            return python_files
            
        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'QUARANTINE', 'archive']]
            
            for file in files:
                if file.endswith('.py') and (max_files is None or len(python_files) < max_files):
                    if not any(skip in file for skip in ['original_', '_original', 'ARCHIVED', 'backup']):
                        python_files.append(Path(root) / file)
        
        return python_files
    
    def _semantic_analysis(self, python_files, base_dir):
        """Semantic analysis dimension using AI-powered intent classification."""
        # STEELCLAD Phase 2: Use extracted semantic analyzer
        return self.semantic_analyzer.analyze_semantic_dimensions(python_files, base_dir)
    
    def _security_analysis(self, python_files, base_dir):
        """Security analysis dimension with vulnerability assessment."""
        # STEELCLAD Phase 2: Use extracted security analyzer
        return self.security_analyzer.analyze_security_dimensions(python_files, base_dir)
    
    def _quality_analysis(self, python_files, base_dir):
        """Quality analysis dimension with maintainability metrics."""
        # STEELCLAD Phase 2: Use extracted quality analyzer
        return self.quality_analyzer.analyze_quality_dimensions(python_files, base_dir)
    
    def _pattern_analysis(self, python_files, base_dir):
        """Pattern recognition dimension using ML-based detection."""
        # STEELCLAD Phase 2: Use extracted pattern analyzer
        return self.pattern_analyzer.analyze_pattern_dimensions(python_files, base_dir)
    
    def _predictive_analysis(self, python_files, base_dir):
        """Predictive analysis dimension for evolution forecasting."""
        # STEELCLAD Phase 2: Use extracted predictive analyzer  
        predictive_results = {
            "evolution_predictions": {},
            "change_impact_radius": {},
            "refactoring_recommendations": {},
            "trend_analysis": {}
        }
        
        base_path = Path(base_dir)
        
        # Analyze evolution patterns using extracted analyzer
        for py_file in python_files:
            try:
                relative_path = str(py_file.relative_to(base_path))
                
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Evolution prediction using extracted analyzer
                evolution = self.predictive_analyzer.predict_evolution(content, relative_path)
                predictive_results["evolution_predictions"][relative_path] = evolution
                
                # Change impact radius using extracted analyzer
                impact = self.predictive_analyzer.calculate_change_impact_radius(content, relative_path)
                predictive_results["change_impact_radius"][relative_path] = impact
                
            except Exception:
                continue
        
        # Generate refactoring recommendations using extracted analyzer
        predictive_results["refactoring_recommendations"] = self.predictive_analyzer.generate_refactoring_recommendations(
            predictive_results["evolution_predictions"],
            predictive_results["change_impact_radius"]
        )
        
        return predictive_results
    
    def _build_multi_layer_graph(self, python_files, base_dir, analysis_results):
        """Build multi-layer graph with all dimensions."""
        graph_data = {
            "nodes": [],
            "links": [],
            "layers": [
                {"id": "functional", "name": "Functional Linkage", "color": "#3b82f6"},
                {"id": "semantic", "name": "Semantic Intent", "color": "#10b981"},
                {"id": "security", "name": "Security Risk", "color": "#ef4444"},
                {"id": "quality", "name": "Quality Metrics", "color": "#f59e0b"},
                {"id": "patterns", "name": "Design Patterns", "color": "#8b5cf6"},
                {"id": "predictive", "name": "Evolution Forecast", "color": "#ec4899"}
            ]
        }
        
        base_path = Path(base_dir)
        
        # Build enhanced nodes
        for py_file in python_files:
            relative_path = str(py_file.relative_to(base_path))
            
            # Get basic linkage info
            basic_info = self._get_basic_file_info(py_file, analysis_results["basic_linkage"])
            
            node = {
                "id": relative_path,
                "name": py_file.name,
                "path": relative_path,
                "layers": {
                    "functional": basic_info,
                    "semantic": analysis_results["semantic_dimensions"]["intent_classifications"].get(relative_path, {}),
                    "security": analysis_results["security_dimensions"]["vulnerability_scores"].get(relative_path, 0),
                    "quality": analysis_results["quality_dimensions"]["complexity_scores"].get(relative_path, 0),
                    "patterns": analysis_results["pattern_dimensions"]["design_patterns"].get(relative_path, []),
                    "predictive": analysis_results["predictive_dimensions"]["evolution_predictions"].get(relative_path, {})
                },
                "composite_score": self._calculate_composite_score(relative_path, analysis_results)
            }
            
            graph_data["nodes"].append(node)
        
        # Build enhanced links with multi-dimensional weights
        graph_data["links"] = self._build_enhanced_links(python_files, base_path, analysis_results)
        
        return graph_data
    
    def _generate_intelligence_summary(self, analysis_results):
        """Generate comprehensive intelligence summary."""
        summary = {
            "total_files_analyzed": len(analysis_results.get("semantic_dimensions", {}).get("intent_classifications", {})),
            "semantic_insights": self._summarize_semantic_insights(analysis_results["semantic_dimensions"]),
            "security_insights": self._summarize_security_insights(analysis_results["security_dimensions"]),
            "quality_insights": self._summarize_quality_insights(analysis_results["quality_dimensions"]),
            "pattern_insights": self._summarize_pattern_insights(analysis_results["pattern_dimensions"]),
            "predictive_insights": self._summarize_predictive_insights(analysis_results["predictive_dimensions"]),
            "top_recommendations": self._generate_top_recommendations(analysis_results),
            "risk_assessment": self._generate_risk_assessment(analysis_results)
        }
        
        return summary
    
    # STEELCLAD Phase 2: Methods moved to extracted analyzer modules
    
    # STEELCLAD Phase 2: Quality and Pattern Analysis Methods moved to extracted modules
    
    # Helper Methods
    def _load_semantic_patterns(self):
        """Load semantic analysis patterns."""
        return {}  # Placeholder
    
    def _load_security_patterns(self):
        """Load security analysis patterns."""
        return {}  # Placeholder
    
    def _load_quality_thresholds(self):
        """Load quality analysis thresholds."""
        return {}  # Placeholder
    
    # STEELCLAD Phase 2: _classify_developer_intent moved to semantic_analyzer module
    
    # STEELCLAD Phase 2: _extract_conceptual_elements moved to semantic_analyzer module

    def _get_vulnerability_weight(self, vuln_type):
        """Get weight for vulnerability type."""
        weights = {
            "sql_injection": 10,
            "xss": 8,
            "hardcoded_secrets": 9,
            "unsafe_deserialization": 9,
            "weak_crypto": 6
        }
        return weights.get(vuln_type, 1)
    
    def _calculate_risk_level(self, score):
        """Calculate risk level from vulnerability score."""
        if score >= 50:
            return "critical"
        elif score >= 20:
            return "high"
        elif score >= 10:
            return "medium"
        else:
            return "low"
    
    def _get_maintainability_level(self, mi):
        """Get maintainability level from index."""
        if mi >= 85:
            return "excellent"
        elif mi >= 65:
            return "good"
        elif mi >= 45:
            return "moderate"
        else:
            return "poor"
    
    def _get_debt_level(self, score):
        """Get technical debt level."""
        if score >= 20:
            return "high"
        elif score >= 10:
            return "medium"
        else:
            return "low"
    
    def _estimate_code_duplication(self, content):
        """Estimate code duplication."""
        lines = content.splitlines()
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) == 0:
            return 0
        return max(0, len(lines) - len(unique_lines))
    
    # Placeholder methods for complex analyses
    def _cluster_by_semantics(self, intent_classifications):
        return {}
    
    def _build_purpose_linkage(self, intent_classifications):
        return {}
    
    def _classify_security_risk(self, vuln_score, patterns):
        return vuln_score.get("risk_level", "low")
    
    def _analyze_security_linkage_impact(self, vuln_scores, risk_classifications):
        return {}
    
    def _correlate_quality_with_linkage(self, complexity_scores, maintainability_index):
        return {}
    
    def _detect_architectural_patterns(self, content):
        return {}
    
    def _cluster_by_patterns(self, design_patterns, arch_patterns):
        return {}
    
    # STEELCLAD Phase 2: Enhanced Predictive Analytics Methods moved to predictive_analyzer module
    
    # STEELCLAD Phase 2: All predictive analysis methods moved to predictive_analyzer module
    
    def _get_basic_file_info(self, py_file, basic_linkage):
        return {}
    
    def _calculate_composite_score(self, file_path, analysis_results):
        return 0
    
    def _build_enhanced_links(self, python_files, base_path, analysis_results):
        return []
    
    def _summarize_semantic_insights(self, semantic_dimensions):
        return {}
    
    def _summarize_security_insights(self, security_dimensions):
        return {}
    
    def _summarize_quality_insights(self, quality_dimensions):
        return {}
    
    def _summarize_pattern_insights(self, pattern_dimensions):
        return {}
    
    def _summarize_predictive_insights(self, predictive_dimensions):
        return {}
    
    def _generate_top_recommendations(self, analysis_results):
        return []
    
    def _generate_risk_assessment(self, analysis_results):
        return {}


def main():
    """Test the enhanced linkage analyzer."""
    analyzer = EnhancedLinkageAnalyzer()
    
    print("Enhanced Multi-Dimensional Linkage Analysis")
    print("=" * 55)
    
    results = analyzer.analyze_codebase()
    
    print(f"\nAnalysis Summary:")
    print(f"Files analyzed: {results['intelligence_summary']['total_files_analyzed']}")
    
    # Save detailed results
    output_file = Path("enhanced_linkage_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()