"""

from .base import FunctionCoverage, ModuleCoverage, CoverageReport

class CodebaseHealthAssessment:
    """
    Comprehensive codebase health assessment using multiple analysis techniques.
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    def assess_codebase_health(self) -> Dict[str, Any]:
        """Perform comprehensive codebase health assessment."""
        assessment = {
            "overall_health_score": 0.0,
            "metrics": {
                "code_quality": self._assess_code_quality(),
                "architectural_integrity": self._assess_architectural_integrity(),
                "test_coverage_health": self._assess_test_coverage_health(),
                "dependency_health": self._assess_dependency_health(),
                "documentation_health": self._assess_documentation_health()
            },
            "recommendations": [],
            "critical_issues": [],
            "health_trends": self._analyze_health_trends()
        }
        
        # Calculate overall health score
        scores = [score for score in assessment["metrics"].values() if isinstance(score, (int, float))]
        assessment["overall_health_score"] = sum(scores) / len(scores) if scores else 0.0
        
        # Generate recommendations
        assessment["recommendations"] = self._generate_health_recommendations(assessment["metrics"])
        
        return assessment
    
    def _assess_code_quality(self) -> float:
        """Assess code quality using multiple metrics."""
        quality_metrics = {
            "average_function_length": 0,
            "cyclomatic_complexity": 0,
            "docstring_coverage": 0,
            "code_duplication": 0
        }
        
        total_functions = 0
        total_lines = 0
        documented_functions = 0
        
        for py_file in self.base_path.rglob("*.py"):
            if self._should_analyze_file(py_file):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if node.end_lineno:
                                func_length = node.end_lineno - node.lineno
                                total_lines += func_length
                            
                            if ast.get_docstring(node):
                                documented_functions += 1
                
                except Exception:
                    continue
        
        if total_functions > 0:
            quality_metrics["average_function_length"] = total_lines / total_functions
            quality_metrics["docstring_coverage"] = (documented_functions / total_functions) * 100
        
        # Simple quality score based on metrics
        quality_score = min(100, 100 - (quality_metrics["average_function_length"] * 2) + quality_metrics["docstring_coverage"])
        return max(0, quality_score)
    
    def _assess_architectural_integrity(self) -> float:
        """Assess architectural integrity."""
        # Simplified architectural assessment
        dependency_mapper = AdvancedDependencyMapper(self.base_path)
        dep_analysis = dependency_mapper.perform_dependency_analysis()
        
        circular_deps = len(dep_analysis["circular_dependencies"])
        orphaned_modules = len(dep_analysis["orphaned_modules"])
        
        # Score based on architectural issues
        integrity_score = 100 - (circular_deps * 10) - (orphaned_modules * 2)
        return max(0, min(100, integrity_score))
    
    def _assess_test_coverage_health(self) -> float:
        """Assess test coverage health."""
        try:
            analyzer = CoverageAnalyzer(str(self.base_path), str(self.base_path / "tests"))
            report = analyzer.run_full_analysis()
            return report.overall_percentage
        except Exception:
            return 0.0
    
    def _assess_dependency_health(self) -> float:
        """Assess dependency health."""
        dependency_mapper = AdvancedDependencyMapper(self.base_path)
        dep_analysis = dependency_mapper.perform_dependency_analysis()
        
        total_deps = sum(len(deps) for deps in dep_analysis["import_dependencies"].values())
        circular_deps = len(dep_analysis["circular_dependencies"])
        
        if total_deps == 0:
            return 100.0
        
        # Score based on dependency issues
        health_score = 100 - (circular_deps / total_deps * 100)
        return max(0, min(100, health_score))
    
    def _assess_documentation_health(self) -> float:
        """Assess documentation health."""
        total_modules = 0
        documented_modules = 0
        
        for py_file in self.base_path.rglob("*.py"):
            if self._should_analyze_file(py_file):
                total_modules += 1
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    
                    if ast.get_docstring(tree):
                        documented_modules += 1
                
                except Exception:
                    continue
        
        return (documented_modules / total_modules * 100) if total_modules > 0 else 0
    
    def _analyze_health_trends(self) -> Dict[str, str]:
        """Analyze health trends over time."""
        # Simplified trend analysis - in real implementation, this would
        # compare with historical data
        return {
            "code_quality_trend": "stable",
            "test_coverage_trend": "improving",
            "dependency_health_trend": "stable",
            "architectural_integrity_trend": "improving"
        }
    
    def _generate_health_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if isinstance(metrics.get("test_coverage_health"), (int, float)):
            if metrics["test_coverage_health"] < 80:
                recommendations.append("Improve test coverage - current coverage below 80%")
        
        if isinstance(metrics.get("code_quality"), (int, float)):
            if metrics["code_quality"] < 70:
                recommendations.append("Improve code quality - consider refactoring large functions and adding documentation")
        
        if isinstance(metrics.get("architectural_integrity"), (int, float)):
            if metrics["architectural_integrity"] < 80:
                recommendations.append("Address architectural issues - resolve circular dependencies and orphaned modules")
        
        return recommendations
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed."""
        if "__pycache__" in str(file_path):
            return False
        if file_path.name.startswith("__") and file_path.name != "__init__.py":
            return False
        return True