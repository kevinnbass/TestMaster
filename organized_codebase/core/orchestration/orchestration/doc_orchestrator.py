"""
Documentation Orchestrator

Unified interface for all documentation modules with smart generation strategies.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .auto_generator import DocumentationAutoGenerator
from .api_spec_builder import APISpecBuilder
from .diagram_creator import DiagramCreator
from .markdown_generator import MarkdownGenerator
from .docstring_analyzer import DocstringAnalyzer
from .changelog_generator import ChangelogGenerator
from .metrics_reporter import MetricsReporter

logger = logging.getLogger(__name__)


@dataclass
class DocStrategy:
    """Documentation generation strategy."""
    name: str
    priority: int
    modules: List[str]
    triggers: List[str]
    quality_threshold: float
    

@dataclass
class DocQualityReport:
    """Documentation quality assessment."""
    overall_score: float
    coverage_percentage: float
    issues_found: int
    recommendations: List[str]
    

class DocumentationOrchestrator:
    """
    Unified documentation orchestrator managing all documentation modules.
    Provides smart generation strategies and quality monitoring.
    """
    
    def __init__(self):
        """Initialize the documentation orchestrator."""
        self.auto_gen = DocumentationAutoGenerator()
        self.api_builder = APISpecBuilder()
        self.diagram_creator = DiagramCreator()
        self.markdown_gen = MarkdownGenerator()
        self.docstring_analyzer = DocstringAnalyzer()
        self.changelog_gen = ChangelogGenerator()
        self.metrics_reporter = MetricsReporter()
        
        self.strategies = self._define_strategies()
        self.generation_history = []
        logger.info("Documentation Orchestrator initialized")
        
    def generate_complete_documentation(self, project_path: str) -> Dict[str, Any]:
        """
        Generate complete documentation suite for a project.
        
        Args:
            project_path: Path to project
            
        Returns:
            Generation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'project_path': project_path,
            'documentation_generated': {},
            'quality_report': None,
            'recommendations': []
        }
        
        # 1. Generate API documentation
        api_endpoints = self.api_builder.scan_directory(project_path)
        if api_endpoints:
            api_spec = self.api_builder.build_openapi_spec(
                title=f"{Path(project_path).name} API",
                version="1.0.0",
                description="Auto-generated API documentation"
            )
            results['documentation_generated']['api_spec'] = api_spec
            
        # 2. Generate architecture diagrams
        arch_analysis = self.diagram_creator.analyze_architecture(project_path)
        mermaid_diagram = self.diagram_creator.generate_mermaid_diagram()
        results['documentation_generated']['architecture'] = {
            'analysis': arch_analysis,
            'mermaid': mermaid_diagram
        }
        
        # 3. Generate module documentation
        module_docs = self.auto_gen.batch_generate(project_path)
        results['documentation_generated']['modules'] = len(module_docs)
        
        # 4. Analyze docstring quality
        quality_analysis = self._analyze_docstring_quality(project_path)
        results['quality_report'] = quality_analysis
        
        # 5. Generate project metrics
        project_metrics = self.metrics_reporter.analyze_project(project_path)
        results['documentation_generated']['metrics'] = project_metrics
        
        # 6. Generate changelog
        commits = self.changelog_gen.parse_commits()
        if commits:
            changelog = self.changelog_gen.generate_changelog()
            results['documentation_generated']['changelog'] = len(commits)
            
        # 7. Generate comprehensive README
        readme_content = self._generate_project_readme(project_path, results)
        results['documentation_generated']['readme'] = readme_content
        
        return results
        
    def monitor_documentation_quality(self, project_path: str) -> DocQualityReport:
        """
        Monitor documentation quality in real-time.
        
        Args:
            project_path: Path to project
            
        Returns:
            Quality report
        """
        issues = []
        scores = []
        
        # Check docstring coverage
        python_files = list(Path(project_path).rglob("*.py"))
        documented_files = 0
        
        for py_file in python_files:
            analysis = self.docstring_analyzer.analyze_file(str(py_file))
            if analysis:
                file_scores = [a.score for a in analysis.values()]
                scores.extend(file_scores)
                if any(score > 50 for score in file_scores):
                    documented_files += 1
                    
        coverage = (documented_files / len(python_files) * 100) if python_files else 0
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # Check for missing documentation
        if coverage < 80:
            issues.append(f"Low documentation coverage: {coverage:.1f}%")
            
        if overall_score < 70:
            issues.append(f"Low docstring quality: {overall_score:.1f}/100")
            
        # Check API documentation
        endpoints = self.api_builder.scan_directory(project_path)
        if endpoints and not any(e.summary for e in endpoints):
            issues.append("API endpoints lack documentation")
            
        recommendations = self._generate_quality_recommendations(coverage, overall_score, issues)
        
        return DocQualityReport(
            overall_score=overall_score,
            coverage_percentage=coverage,
            issues_found=len(issues),
            recommendations=recommendations
        )
        
    def auto_update_documentation(self, project_path: str, changed_files: List[str]) -> Dict[str, Any]:
        """
        Automatically update documentation for changed files.
        
        Args:
            project_path: Project path
            changed_files: List of changed files
            
        Returns:
            Update results
        """
        updates = {
            'timestamp': datetime.now().isoformat(),
            'files_processed': [],
            'documentation_updated': []
        }
        
        for file_path in changed_files:
            if file_path.endswith('.py'):
                # Update module documentation
                doc = self.auto_gen.generate_documentation(file_path)
                if doc:
                    updates['files_processed'].append(file_path)
                    updates['documentation_updated'].append('module_docs')
                    
                # Update API documentation if it's an API file
                if any(keyword in file_path.lower() for keyword in ['api', 'route', 'endpoint']):
                    endpoints = self.api_builder.analyze_flask_routes(file_path)
                    if endpoints:
                        updates['documentation_updated'].append('api_spec')
                        
        # Update architecture diagram if significant changes
        if len(changed_files) > 5:
            self.diagram_creator.analyze_architecture(project_path)
            updates['documentation_updated'].append('architecture_diagram')
            
        return updates
        
    def get_documentation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive documentation metrics."""
        return {
            'generators': {
                'auto_generator': self.auto_gen.get_metrics(),
                'api_endpoints': len(self.api_builder.endpoints),
                'architecture_components': len(self.diagram_creator.components)
            },
            'generation_history': len(self.generation_history),
            'quality_monitoring': True,
            'strategies_defined': len(self.strategies)
        }
        
    # Private methods
    def _define_strategies(self) -> List[DocStrategy]:
        """Define documentation generation strategies."""
        return [
            DocStrategy(
                name="comprehensive",
                priority=1,
                modules=["auto_gen", "api_builder", "diagram_creator", "markdown_gen"],
                triggers=["new_project", "major_release"],
                quality_threshold=90.0
            ),
            DocStrategy(
                name="incremental",
                priority=2,
                modules=["auto_gen", "docstring_analyzer"],
                triggers=["file_change", "commit"],
                quality_threshold=70.0
            ),
            DocStrategy(
                name="api_focused",
                priority=3,
                modules=["api_builder", "markdown_gen"],
                triggers=["api_change", "endpoint_added"],
                quality_threshold=80.0
            )
        ]
        
    def _analyze_docstring_quality(self, project_path: str) -> Dict[str, Any]:
        """Analyze docstring quality across project."""
        total_score = 0
        file_count = 0
        issues = []
        
        for py_file in Path(project_path).rglob("*.py"):
            analysis = self.docstring_analyzer.analyze_file(str(py_file))
            if analysis:
                file_scores = [a.score for a in analysis.values()]
                total_score += sum(file_scores)
                file_count += len(file_scores)
                
                # Collect critical issues
                for name, result in analysis.items():
                    critical_issues = [i for i in result.issues if i.severity == "error"]
                    if critical_issues:
                        issues.extend(critical_issues)
                        
        avg_score = total_score / file_count if file_count > 0 else 0
        
        return {
            'average_score': avg_score,
            'files_analyzed': file_count,
            'critical_issues': len(issues),
            'quality_grade': self._calculate_quality_grade(avg_score)
        }
        
    def _generate_project_readme(self, project_path: str, results: Dict) -> str:
        """Generate comprehensive project README."""
        project_name = Path(project_path).name
        
        # Extract features from analysis
        features = []
        if 'api_spec' in results['documentation_generated']:
            features.append("REST API with OpenAPI specification")
        if results['documentation_generated'].get('modules', 0) > 0:
            features.append(f"Modular architecture with {results['documentation_generated']['modules']} documented modules")
        if 'architecture' in results['documentation_generated']:
            features.append("Interactive architecture diagrams")
            
        self.markdown_gen.add_badge("Documentation", "Auto-Generated", "blue")
        self.markdown_gen.add_badge("Quality", 
                                   results['quality_report']['quality_grade'], 
                                   "green" if results['quality_report']['average_score'] > 80 else "yellow")
        
        installation = f"pip install {project_name.lower()}"
        usage = f"from {project_name.lower()} import main\nmain.run()"
        
        return self.markdown_gen.generate_readme(
            project_name=project_name,
            description=f"Auto-documented {project_name} project",
            installation=installation,
            usage=usage,
            features=features
        )
        
    def _generate_quality_recommendations(self, coverage: float, score: float, issues: List) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if coverage < 50:
            recommendations.append("Critical: Add docstrings to at least 50% of functions and classes")
        elif coverage < 80:
            recommendations.append("Improve documentation coverage to 80%+")
            
        if score < 50:
            recommendations.append("Critical: Improve docstring quality with proper descriptions and parameters")
        elif score < 70:
            recommendations.append("Enhance docstring completeness and style compliance")
            
        if issues:
            recommendations.append("Address critical documentation issues found in analysis")
            
        recommendations.append("Consider adding usage examples to key modules")
        recommendations.append("Implement automated documentation updates in CI/CD")
        
        return recommendations
        
    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate quality grade from score."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "F"