"""
Code Metrics Reporter

Generates comprehensive code quality metrics and reports.
"""

import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FileMetrics:
    """Metrics for a single file."""
    file_path: str
    lines_total: int
    lines_code: int
    lines_comment: int
    lines_blank: int
    complexity: int
    functions: int
    classes: int
    imports: int
    

@dataclass
class QualityMetrics:
    """Code quality metrics."""
    maintainability_index: float  # 0-100
    cyclomatic_complexity: int
    cognitive_complexity: int
    halstead_metrics: Dict[str, float]
    code_duplication: float  # percentage
    

class MetricsReporter:
    """
    Generates comprehensive code metrics and quality reports.
    Calculates complexity, maintainability, and other quality indicators.
    """
    
    def __init__(self):
        """Initialize the metrics reporter."""
        self.file_metrics = []
        self.project_metrics = {}
        logger.info("Metrics Reporter initialized")
        
    def analyze_file(self, file_path: str) -> FileMetrics:
        """
        Analyze a single file for metrics.
        
        Args:
            file_path: Path to file
            
        Returns:
            File metrics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                
            tree = ast.parse(content)
            
            metrics = FileMetrics(
                file_path=file_path,
                lines_total=len(lines),
                lines_code=self._count_code_lines(lines),
                lines_comment=self._count_comment_lines(lines),
                lines_blank=self._count_blank_lines(lines),
                complexity=self._calculate_complexity(tree),
                functions=self._count_functions(tree),
                classes=self._count_classes(tree),
                imports=self._count_imports(tree)
            )
            
            self.file_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
            
    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """
        Analyze entire project for metrics.
        
        Args:
            project_path: Path to project
            
        Returns:
            Project-wide metrics
        """
        project = Path(project_path)
        
        # Analyze all Python files
        for py_file in project.rglob("*.py"):
            if '__pycache__' not in str(py_file):
                self.analyze_file(str(py_file))
                
        # Calculate aggregate metrics
        self.project_metrics = self._calculate_project_metrics()
        
        return self.project_metrics
        
    def _calculate_project_metrics(self) -> Dict[str, Any]:
        """Calculate project-wide metrics."""
        if not self.file_metrics:
            return {}
            
        total_lines = sum(m.lines_total for m in self.file_metrics)
        total_code = sum(m.lines_code for m in self.file_metrics)
        total_comments = sum(m.lines_comment for m in self.file_metrics)
        total_complexity = sum(m.complexity for m in self.file_metrics)
        
        return {
            'files': len(self.file_metrics),
            'total_lines': total_lines,
            'code_lines': total_code,
            'comment_lines': total_comments,
            'blank_lines': sum(m.lines_blank for m in self.file_metrics),
            'comment_ratio': (total_comments / total_code * 100) if total_code > 0 else 0,
            'avg_file_size': total_lines / len(self.file_metrics),
            'total_functions': sum(m.functions for m in self.file_metrics),
            'total_classes': sum(m.classes for m in self.file_metrics),
            'avg_complexity': total_complexity / len(self.file_metrics),
            'max_complexity': max(m.complexity for m in self.file_metrics)
        }
        
    def calculate_quality_metrics(self, file_path: str) -> QualityMetrics:
        """
        Calculate code quality metrics.
        
        Args:
            file_path: Path to file
            
        Returns:
            Quality metrics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
            cyclomatic = self._calculate_complexity(tree)
            cognitive = self._calculate_cognitive_complexity(tree)
            halstead = self._calculate_halstead_metrics(tree)
            
            # Maintainability Index formula
            # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
            loc = len(content.splitlines())
            hv = halstead.get('volume', 1)
            mi = max(0, min(100, 171 - 5.2 * (hv ** 0.5) - 0.23 * cyclomatic - 16.2 * (loc ** 0.5)))
            
            return QualityMetrics(
                maintainability_index=mi,
                cyclomatic_complexity=cyclomatic,
                cognitive_complexity=cognitive,
                halstead_metrics=halstead,
                code_duplication=self._detect_duplication(content)
            )
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return None
            
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
        
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity."""
        complexity = 0
        nesting_level = 0
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting = 0
                
            def visit_If(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
                
            def visit_For(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
                
            def visit_While(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
                
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return visitor.complexity
        
    def _calculate_halstead_metrics(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate Halstead complexity metrics."""
        operators = set()
        operands = set()
        total_operators = 0
        total_operands = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
                operators.add(type(node).__name__)
                total_operators += 1
            elif isinstance(node, ast.Name):
                operands.add(node.id)
                total_operands += 1
                
        n1 = len(operators)  # Unique operators
        n2 = len(operands)   # Unique operands
        N1 = total_operators # Total operators
        N2 = total_operands  # Total operands
        
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * (vocabulary ** 0.5) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = volume * difficulty
        
        return {
            'vocabulary': vocabulary,
            'length': length,
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort
        }
        
    def _detect_duplication(self, content: str) -> float:
        """Detect code duplication percentage."""
        lines = content.splitlines()
        unique_lines = set(line.strip() for line in lines if line.strip())
        
        if not lines:
            return 0.0
            
        duplication = (1 - len(unique_lines) / len(lines)) * 100
        return max(0, duplication)
        
    def generate_report(self, format: str = "markdown") -> str:
        """
        Generate metrics report.
        
        Args:
            format: Report format (markdown, json, html)
            
        Returns:
            Formatted report
        """
        if format == "markdown":
            return self._generate_markdown_report()
        elif format == "json":
            return self._generate_json_report()
        else:
            return self._generate_markdown_report()
            
    def _generate_markdown_report(self) -> str:
        """Generate markdown format report."""
        report = [
            "# Code Metrics Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Project Summary",
            ""
        ]
        
        if self.project_metrics:
            report.extend([
                f"- **Files Analyzed:** {self.project_metrics['files']}",
                f"- **Total Lines:** {self.project_metrics['total_lines']:,}",
                f"- **Code Lines:** {self.project_metrics['code_lines']:,}",
                f"- **Comment Ratio:** {self.project_metrics['comment_ratio']:.1f}%",
                f"- **Average Complexity:** {self.project_metrics['avg_complexity']:.1f}",
                "",
                "## Top Complex Files",
                ""
            ])
            
            # Sort by complexity
            complex_files = sorted(self.file_metrics, key=lambda x: x.complexity, reverse=True)[:10]
            
            for i, file_metric in enumerate(complex_files, 1):
                report.append(f"{i}. {Path(file_metric.file_path).name} - Complexity: {file_metric.complexity}")
                
        return "\n".join(report)
        
    def _generate_json_report(self) -> str:
        """Generate JSON format report."""
        import json
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_metrics': self.project_metrics,
            'file_metrics': [
                {
                    'file': m.file_path,
                    'lines': m.lines_total,
                    'complexity': m.complexity
                }
                for m in self.file_metrics
            ]
        }
        
        return json.dumps(report, indent=2)
        
    # Helper methods
    def _count_code_lines(self, lines: List[str]) -> int:
        """Count lines containing code."""
        return sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        
    def _count_comment_lines(self, lines: List[str]) -> int:
        """Count comment lines."""
        return sum(1 for line in lines if line.strip().startswith('#'))
        
    def _count_blank_lines(self, lines: List[str]) -> int:
        """Count blank lines."""
        return sum(1 for line in lines if not line.strip())
        
    def _count_functions(self, tree: ast.AST) -> int:
        """Count functions in AST."""
        return sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
    def _count_classes(self, tree: ast.AST) -> int:
        """Count classes in AST."""
        return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        
    def _count_imports(self, tree: ast.AST) -> int:
        """Count imports in AST."""
        return sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))