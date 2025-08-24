"""
Unified Coverage Analysis Interface
"""

from .base import FunctionCoverage, ModuleCoverage, CoverageReport
from .analyzer import CoverageAnalyzer
from .codebase_analyzer import ComprehensiveCodebaseAnalyzer
from .dependency_mapper import AdvancedDependencyMapper
from .health_assessment import CodebaseHealthAssessment


class UnifiedCoverageAnalyzer:
    """
    Unified interface for all coverage analysis functionality.
    """
    
    def __init__(self):
        """Initialize the unified coverage analyzer."""
        self.analyzer = CoverageAnalyzer()
        self.codebase_analyzer = ComprehensiveCodebaseAnalyzer()
        self.dependency_mapper = AdvancedDependencyMapper()
        self.health_assessment = CodebaseHealthAssessment()
    
    def analyze_coverage(self, project_path: str = ".") -> CoverageReport:
        """Run comprehensive coverage analysis."""
        return self.analyzer.analyze_project_coverage(project_path)
    
    def analyze_codebase(self, project_path: str = "."):
        """Run comprehensive codebase analysis."""
        return self.codebase_analyzer.analyze_codebase(project_path)
    
    def map_dependencies(self, project_path: str = "."):
        """Map project dependencies."""
        return self.dependency_mapper.analyze_dependencies(project_path)
    
    def assess_health(self, project_path: str = "."):
        """Assess codebase health."""
        return self.health_assessment.assess_project_health(project_path)
    
    def generate_comprehensive_report(self, project_path: str = "."):
        """Generate a comprehensive analysis report."""
        coverage = self.analyze_coverage(project_path)
        codebase = self.analyze_codebase(project_path)
        dependencies = self.map_dependencies(project_path)
        health = self.assess_health(project_path)
        
        return {
            'coverage': coverage,
            'codebase': codebase, 
            'dependencies': dependencies,
            'health': health,
            'timestamp': coverage.timestamp
        }
