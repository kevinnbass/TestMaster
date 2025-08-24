"""
Documentation Completeness Checker

Comprehensive checker for documentation completeness across codebases.
Analyzes coverage, identifies gaps, and provides improvement recommendations.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CompletenessLevel(Enum):
    """Documentation completeness levels."""
    NONE = "none"
    MINIMAL = "minimal" 
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    EXCELLENT = "excellent"


@dataclass
class DocumentationGap:
    """Represents a documentation gap."""
    item_type: str  # "function", "class", "module"
    item_name: str
    file_path: str
    gap_type: str  # "missing", "incomplete", "outdated"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    suggestions: List[str]


@dataclass
class CompletenessReport:
    """Comprehensive documentation completeness report."""
    overall_score: float  # 0.0 to 1.0
    coverage_by_type: Dict[str, float]
    total_items: int
    documented_items: int
    gaps: List[DocumentationGap]
    recommendations: List[str]
    completeness_level: CompletenessLevel


class CompletenessChecker:
    """
    Documentation completeness checker.
    
    Analyzes Python codebases to identify documentation gaps and provide
    comprehensive coverage analysis with actionable recommendations.
    """
    
    def __init__(self):
        """Initialize the completeness checker."""
        self.required_sections = {
            "function": ["description", "parameters", "returns"],
            "class": ["description", "attributes", "methods"],
            "module": ["description", "functions", "classes"]
        }
        
    def check_project_completeness(self, project_path: str) -> CompletenessReport:
        """
        Check documentation completeness for an entire project.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            CompletenessReport: Comprehensive completeness analysis
        """
        logger.info(f"Checking documentation completeness for: {project_path}")
        
        # Find all Python files
        python_files = self._find_python_files(project_path)
        
        # Analyze each file
        all_gaps = []
        total_items = 0
        documented_items = 0
        coverage_by_type = {"functions": 0.0, "classes": 0.0, "modules": 0.0}
        
        for file_path in python_files:
            try:
                file_report = self.check_file_completeness(file_path)
                all_gaps.extend(file_report.gaps)
                total_items += file_report.total_items
                documented_items += file_report.documented_items
                
                # Aggregate coverage by type
                for item_type, coverage in file_report.coverage_by_type.items():
                    if item_type in coverage_by_type:
                        coverage_by_type[item_type] += coverage
                        
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
        
        # Calculate overall metrics
        overall_score = documented_items / total_items if total_items > 0 else 0.0
        
        # Normalize coverage by type
        file_count = len(python_files)
        if file_count > 0:
            for item_type in coverage_by_type:
                coverage_by_type[item_type] /= file_count
        
        # Determine completeness level
        completeness_level = self._determine_completeness_level(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_project_recommendations(all_gaps, coverage_by_type)
        
        report = CompletenessReport(
            overall_score=overall_score,
            coverage_by_type=coverage_by_type,
            total_items=total_items,
            documented_items=documented_items,
            gaps=all_gaps,
            recommendations=recommendations,
            completeness_level=completeness_level
        )
        
        logger.info(f"Project completeness: {overall_score:.2f} ({completeness_level.value})")
        return report
        
    def check_file_completeness(self, file_path: str) -> CompletenessReport:
        """
        Check documentation completeness for a single file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            CompletenessReport: File completeness analysis
        """
        logger.debug(f"Checking file completeness: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return self._empty_report()
        
        gaps = []
        total_items = 0
        documented_items = 0
        
        # Check module-level documentation
        module_gap = self._check_module_documentation(tree, file_path)
        if module_gap:
            gaps.append(module_gap)
        else:
            documented_items += 1
        total_items += 1
        
        # Check functions
        functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        func_documented = 0
        
        for func_node in functions:
            if not func_node.name.startswith('_'):  # Skip private functions
                gap = self._check_function_documentation(func_node, file_path)
                if gap:
                    gaps.append(gap)
                else:
                    func_documented += 1
                total_items += 1
        
        # Check classes
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        class_documented = 0
        
        for class_node in classes:
            if not class_node.name.startswith('_'):  # Skip private classes
                gap = self._check_class_documentation(class_node, file_path)
                if gap:
                    gaps.append(gap)
                else:
                    class_documented += 1
                total_items += 1
        
        documented_items += func_documented + class_documented
        
        # Calculate coverage by type
        coverage_by_type = {
            "functions": func_documented / len(functions) if functions else 1.0,
            "classes": class_documented / len(classes) if classes else 1.0,
            "modules": 1.0 if not module_gap else 0.0
        }
        
        overall_score = documented_items / total_items if total_items > 0 else 0.0
        completeness_level = self._determine_completeness_level(overall_score)
        
        return CompletenessReport(
            overall_score=overall_score,
            coverage_by_type=coverage_by_type,
            total_items=total_items,
            documented_items=documented_items,
            gaps=gaps,
            recommendations=self._generate_file_recommendations(gaps),
            completeness_level=completeness_level
        )
        
    def check_function_completeness(self, file_path: str, function_name: str) -> Optional[DocumentationGap]:
        """
        Check documentation completeness for a specific function.
        
        Args:
            file_path: Path to the file containing the function
            function_name: Name of the function
            
        Returns:
            Optional[DocumentationGap]: Gap if documentation is incomplete
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return self._check_function_documentation(node, file_path)
        
        return DocumentationGap(
            item_type="function",
            item_name=function_name,
            file_path=file_path,
            gap_type="missing",
            severity="critical",
            description=f"Function {function_name} not found",
            suggestions=["Verify function name and file path"]
        )
        
    def check_class_completeness(self, file_path: str, class_name: str) -> Optional[DocumentationGap]:
        """
        Check documentation completeness for a specific class.
        
        Args:
            file_path: Path to the file containing the class
            class_name: Name of the class
            
        Returns:
            Optional[DocumentationGap]: Gap if documentation is incomplete
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return self._check_class_documentation(node, file_path)
        
        return DocumentationGap(
            item_type="class",
            item_name=class_name,
            file_path=file_path,
            gap_type="missing",
            severity="critical",
            description=f"Class {class_name} not found",
            suggestions=["Verify class name and file path"]
        )
        
    def _check_module_documentation(self, tree: ast.AST, file_path: str) -> Optional[DocumentationGap]:
        """Check if module has adequate documentation."""
        # Check for module docstring
        if (tree.body and 
            isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Constant) and
            isinstance(tree.body[0].value.value, str)):
            
            docstring = tree.body[0].value.value
            
            # Check docstring quality
            if len(docstring.strip()) < 10:
                return DocumentationGap(
                    item_type="module",
                    item_name=Path(file_path).stem,
                    file_path=file_path,
                    gap_type="incomplete",
                    severity="medium",
                    description="Module docstring is too brief",
                    suggestions=[
                        "Expand module docstring with purpose and usage",
                        "Document key functions and classes",
                        "Add examples if appropriate"
                    ]
                )
            
            return None  # Module has adequate documentation
        
        return DocumentationGap(
            item_type="module",
            item_name=Path(file_path).stem,
            file_path=file_path,
            gap_type="missing",
            severity="high",
            description="Module missing docstring",
            suggestions=[
                "Add module-level docstring explaining purpose",
                "Document main functionality and usage",
                "Include import examples if needed"
            ]
        )
        
    def _check_function_documentation(self, func_node: ast.FunctionDef, 
                                    file_path: str) -> Optional[DocumentationGap]:
        """Check if function has adequate documentation."""
        # Check for function docstring
        if (func_node.body and 
            isinstance(func_node.body[0], ast.Expr) and 
            isinstance(func_node.body[0].value, ast.Constant) and
            isinstance(func_node.body[0].value.value, str)):
            
            docstring = func_node.body[0].value.value
            
            # Analyze docstring completeness
            gaps = []
            
            # Check for parameter documentation
            if func_node.args.args:
                param_names = [arg.arg for arg in func_node.args.args if arg.arg != 'self']
                missing_params = []
                
                for param in param_names:
                    if param not in docstring:
                        missing_params.append(param)
                
                if missing_params:
                    gaps.append(f"Missing parameter documentation: {', '.join(missing_params)}")
            
            # Check for return documentation
            if func_node.returns or "return" in ast.unparse(func_node):
                if not any(keyword in docstring.lower() for keyword in ["return", "yield"]):
                    gaps.append("Missing return value documentation")
            
            # Check for exception documentation
            # Simple heuristic: look for raise statements
            for node in ast.walk(func_node):
                if isinstance(node, ast.Raise):
                    if not any(keyword in docstring.lower() for keyword in ["raise", "exception", "error"]):
                        gaps.append("Missing exception documentation")
                    break
            
            if gaps:
                return DocumentationGap(
                    item_type="function",
                    item_name=func_node.name,
                    file_path=file_path,
                    gap_type="incomplete",
                    severity="medium",
                    description=f"Function docstring incomplete: {'; '.join(gaps)}",
                    suggestions=[
                        "Document all parameters with types",
                        "Document return value and type",
                        "Document possible exceptions",
                        "Add usage examples if complex"
                    ]
                )
            
            return None  # Function has adequate documentation
        
        return DocumentationGap(
            item_type="function",
            item_name=func_node.name,
            file_path=file_path,
            gap_type="missing",
            severity="high",
            description="Function missing docstring",
            suggestions=[
                "Add comprehensive docstring with description",
                "Document parameters and return value",
                "Include usage example if appropriate"
            ]
        )
        
    def _check_class_documentation(self, class_node: ast.ClassDef, 
                                 file_path: str) -> Optional[DocumentationGap]:
        """Check if class has adequate documentation."""
        # Check for class docstring
        if (class_node.body and 
            isinstance(class_node.body[0], ast.Expr) and 
            isinstance(class_node.body[0].value, ast.Constant) and
            isinstance(class_node.body[0].value.value, str)):
            
            docstring = class_node.body[0].value.value
            
            # Analyze docstring completeness
            gaps = []
            
            # Check for attribute documentation
            # Look for attributes in __init__ method
            init_method = None
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                    init_method = node
                    break
            
            if init_method:
                # Find self.attribute assignments
                attributes = set()
                for node in ast.walk(init_method):
                    if (isinstance(node, ast.Assign) and 
                        len(node.targets) == 1 and
                        isinstance(node.targets[0], ast.Attribute) and
                        isinstance(node.targets[0].value, ast.Name) and
                        node.targets[0].value.id == "self"):
                        attributes.add(node.targets[0].attr)
                
                # Check if attributes are documented
                if attributes:
                    missing_attrs = [attr for attr in attributes if attr not in docstring]
                    if missing_attrs:
                        gaps.append(f"Missing attribute documentation: {', '.join(missing_attrs)}")
            
            # Check for method overview
            methods = [node.name for node in class_node.body 
                      if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')]
            
            if methods and not any(method in docstring for method in methods[:3]):
                gaps.append("Missing key methods overview")
            
            if gaps:
                return DocumentationGap(
                    item_type="class",
                    item_name=class_node.name,
                    file_path=file_path,
                    gap_type="incomplete",
                    severity="medium",
                    description=f"Class docstring incomplete: {'; '.join(gaps)}",
                    suggestions=[
                        "Document class purpose and responsibilities",
                        "Document key attributes",
                        "Provide usage examples",
                        "Explain relationships with other classes"
                    ]
                )
            
            return None  # Class has adequate documentation
        
        return DocumentationGap(
            item_type="class",
            item_name=class_node.name,
            file_path=file_path,
            gap_type="missing",
            severity="high",
            description="Class missing docstring",
            suggestions=[
                "Add comprehensive class docstring",
                "Document class purpose and responsibilities",
                "Document key attributes and methods",
                "Include usage examples"
            ]
        )
        
    def _find_python_files(self, project_path: str) -> List[str]:
        """Find all Python files in a project."""
        python_files = []
        project_dir = Path(project_path)
        
        for file_path in project_dir.rglob("*.py"):
            # Skip hidden directories and common non-source directories
            if not any(part.startswith('.') for part in file_path.parts):
                if not any(skip_dir in file_path.parts 
                          for skip_dir in ['__pycache__', 'venv', 'env', 'node_modules']):
                    python_files.append(str(file_path))
        
        return python_files
        
    def _determine_completeness_level(self, score: float) -> CompletenessLevel:
        """Determine completeness level from score."""
        if score >= 0.9:
            return CompletenessLevel.EXCELLENT
        elif score >= 0.7:
            return CompletenessLevel.COMPREHENSIVE
        elif score >= 0.5:
            return CompletenessLevel.BASIC
        elif score >= 0.2:
            return CompletenessLevel.MINIMAL
        else:
            return CompletenessLevel.NONE
            
    def _generate_project_recommendations(self, gaps: List[DocumentationGap], 
                                        coverage_by_type: Dict[str, float]) -> List[str]:
        """Generate project-level recommendations."""
        recommendations = []
        
        # Priority recommendations based on coverage
        if coverage_by_type.get("modules", 0) < 0.5:
            recommendations.append("PRIORITY: Add module-level docstrings to explain file purposes")
        
        if coverage_by_type.get("functions", 0) < 0.5:
            recommendations.append("PRIORITY: Document function parameters and return values")
        
        if coverage_by_type.get("classes", 0) < 0.5:
            recommendations.append("PRIORITY: Add class docstrings explaining responsibilities")
        
        # Gap-based recommendations
        critical_gaps = [gap for gap in gaps if gap.severity == "critical"]
        if critical_gaps:
            recommendations.append(f"Address {len(critical_gaps)} critical documentation gaps immediately")
        
        high_gaps = [gap for gap in gaps if gap.severity == "high"]
        if high_gaps:
            recommendations.append(f"Address {len(high_gaps)} high-priority documentation gaps")
        
        # Specific recommendations
        missing_gaps = [gap for gap in gaps if gap.gap_type == "missing"]
        if missing_gaps:
            recommendations.append("Focus on adding missing docstrings before improving existing ones")
        
        incomplete_gaps = [gap for gap in gaps if gap.gap_type == "incomplete"]
        if incomplete_gaps:
            recommendations.append("Enhance existing docstrings with complete parameter and return documentation")
        
        if not recommendations:
            recommendations.append("Excellent documentation coverage! Consider adding more examples and cross-references")
        
        return recommendations
        
    def _generate_file_recommendations(self, gaps: List[DocumentationGap]) -> List[str]:
        """Generate file-level recommendations."""
        if not gaps:
            return ["Documentation is complete for this file"]
        
        recommendations = []
        
        # Group gaps by type
        missing_gaps = [gap for gap in gaps if gap.gap_type == "missing"]
        incomplete_gaps = [gap for gap in gaps if gap.gap_type == "incomplete"]
        
        if missing_gaps:
            recommendations.append(f"Add docstrings to {len(missing_gaps)} items")
        
        if incomplete_gaps:
            recommendations.append(f"Complete documentation for {len(incomplete_gaps)} items")
        
        # Specific suggestions from gaps
        all_suggestions = []
        for gap in gaps:
            all_suggestions.extend(gap.suggestions)
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        for suggestion in all_suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)
        
        recommendations.extend(unique_suggestions[:5])  # Limit to top 5
        
        return recommendations
        
    def _empty_report(self) -> CompletenessReport:
        """Return an empty completeness report."""
        return CompletenessReport(
            overall_score=0.0,
            coverage_by_type={"functions": 0.0, "classes": 0.0, "modules": 0.0},
            total_items=0,
            documented_items=0,
            gaps=[],
            recommendations=["Unable to analyze file due to syntax errors"],
            completeness_level=CompletenessLevel.NONE
        )
        
    def export_report(self, report: CompletenessReport, output_path: str, 
                     format: str = "markdown") -> str:
        """
        Export completeness report to file.
        
        Args:
            report: Completeness report to export
            output_path: Output file path
            format: Export format ("markdown", "json", "html")
            
        Returns:
            str: Path to exported file
        """
        if format == "markdown":
            content = self._export_markdown_report(report)
        elif format == "json":
            import json
            content = json.dumps(report.__dict__, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Exported completeness report to: {output_path}")
        return output_path
        
    def _export_markdown_report(self, report: CompletenessReport) -> str:
        """Export report as markdown."""
        lines = [
            "# Documentation Completeness Report",
            "",
            f"**Overall Score**: {report.overall_score:.2f} ({report.completeness_level.value.title()})",
            f"**Items Documented**: {report.documented_items}/{report.total_items}",
            "",
            "## Coverage by Type",
            ""
        ]
        
        for item_type, coverage in report.coverage_by_type.items():
            lines.append(f"- **{item_type.title()}**: {coverage:.2f}")
        
        lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        for recommendation in report.recommendations:
            lines.append(f"- {recommendation}")
        
        if report.gaps:
            lines.extend([
                "",
                "## Documentation Gaps",
                ""
            ])
            
            for gap in report.gaps:
                lines.extend([
                    f"### {gap.item_name} ({gap.item_type})",
                    "",
                    f"**File**: {gap.file_path}",
                    f"**Issue**: {gap.description}",
                    f"**Severity**: {gap.severity}",
                    "",
                    "**Suggestions**:",
                ])
                
                for suggestion in gap.suggestions:
                    lines.append(f"- {suggestion}")
                
                lines.append("")
        
        return "\n".join(lines)