"""
Docstring Quality Analyzer

Analyzes and scores docstring quality, completeness, and style compliance.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DocstringStyle(Enum):
    """Supported docstring styles."""
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    EPYDOC = "epydoc"
    

@dataclass
class DocstringIssue:
    """Represents a docstring quality issue."""
    type: str
    severity: str  # error, warning, info
    line: int
    message: str
    suggestion: str
    

@dataclass
class DocstringAnalysis:
    """Complete docstring analysis results."""
    score: float  # 0-100
    style: DocstringStyle
    issues: List[DocstringIssue]
    metrics: Dict[str, Any]
    suggestions: List[str]
    

class DocstringAnalyzer:
    """
    Analyzes docstring quality, completeness, and style compliance.
    Provides scoring and improvement suggestions.
    """
    
    def __init__(self, style: DocstringStyle = DocstringStyle.GOOGLE):
        """
        Initialize the docstring analyzer.
        
        Args:
            style: Expected docstring style
        """
        self.style = style
        self.total_analyzed = 0
        self.patterns = self._load_style_patterns()
        logger.info(f"Docstring Analyzer initialized with {style.value} style")
        
    def analyze_file(self, file_path: str) -> Dict[str, DocstringAnalysis]:
        """
        Analyze all docstrings in a file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Analysis results for each function/class
        """
        results = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    analysis = self._analyze_node(node)
                    results[node.name] = analysis
                    self.total_analyzed += 1
                    
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            
        return results
        
    def _analyze_node(self, node: ast.AST) -> DocstringAnalysis:
        """
        Analyze a single function or class node.
        
        Args:
            node: AST node to analyze
            
        Returns:
            Docstring analysis
        """
        docstring = ast.get_docstring(node)
        issues = []
        metrics = {}
        suggestions = []
        
        if not docstring:
            issues.append(DocstringIssue(
                type="missing",
                severity="error",
                line=node.lineno,
                message="Missing docstring",
                suggestion="Add a docstring describing the purpose"
            ))
            score = 0
        else:
            # Analyze completeness
            issues.extend(self._check_completeness(node, docstring))
            
            # Check style compliance
            issues.extend(self._check_style(docstring))
            
            # Calculate metrics
            metrics = self._calculate_metrics(docstring)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(node, docstring, issues)
            
            # Calculate score
            score = self._calculate_score(issues, metrics)
            
        return DocstringAnalysis(
            score=score,
            style=self.style,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions
        )
        
    def _check_completeness(self, node: ast.AST, docstring: str) -> List[DocstringIssue]:
        """Check docstring completeness."""
        issues = []
        
        if isinstance(node, ast.FunctionDef):
            # Check for parameter documentation
            for arg in node.args.args:
                if arg.arg != 'self' and arg.arg not in docstring:
                    issues.append(DocstringIssue(
                        type="missing_param",
                        severity="warning",
                        line=node.lineno,
                        message=f"Parameter '{arg.arg}' not documented",
                        suggestion=f"Add documentation for parameter '{arg.arg}'"
                    ))
                    
            # Check for return documentation
            if node.returns and 'return' not in docstring.lower():
                issues.append(DocstringIssue(
                    type="missing_return",
                    severity="warning",
                    line=node.lineno,
                    message="Return value not documented",
                    suggestion="Add Returns section"
                ))
                
            # Check for raises documentation
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
            if has_raise and 'raise' not in docstring.lower():
                issues.append(DocstringIssue(
                    type="missing_raises",
                    severity="info",
                    line=node.lineno,
                    message="Exceptions not documented",
                    suggestion="Add Raises section for exceptions"
                ))
                
        return issues
        
    def _check_style(self, docstring: str) -> List[DocstringIssue]:
        """Check docstring style compliance."""
        issues = []
        
        if self.style == DocstringStyle.GOOGLE:
            # Check for Google style sections
            sections = ['Args:', 'Returns:', 'Raises:', 'Yields:', 'Note:', 'Example:']
            
            # Check first line
            lines = docstring.split('\n')
            if lines[0][-1] == '.':
                issues.append(DocstringIssue(
                    type="style",
                    severity="info",
                    line=0,
                    message="First line should not end with period in Google style",
                    suggestion="Remove period from first line"
                ))
                
        elif self.style == DocstringStyle.NUMPY:
            # Check for NumPy style sections
            sections = ['Parameters', 'Returns', 'Raises', 'See Also', 'Notes', 'Examples']
            
        return issues
        
    def _calculate_metrics(self, docstring: str) -> Dict[str, Any]:
        """Calculate docstring metrics."""
        lines = docstring.split('\n')
        words = docstring.split()
        
        return {
            'length': len(docstring),
            'lines': len(lines),
            'words': len(words),
            'avg_line_length': len(docstring) / max(len(lines), 1),
            'has_examples': 'example' in docstring.lower() or '>>>' in docstring,
            'has_params': 'arg' in docstring.lower() or 'param' in docstring.lower(),
            'has_returns': 'return' in docstring.lower(),
            'has_raises': 'raise' in docstring.lower() or 'except' in docstring.lower()
        }
        
    def _calculate_score(self, issues: List[DocstringIssue], metrics: Dict) -> float:
        """Calculate docstring quality score."""
        score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == "error":
                score -= 20
            elif issue.severity == "warning":
                score -= 10
            elif issue.severity == "info":
                score -= 5
                
        # Bonus for good practices
        if metrics.get('has_examples'):
            score += 5
        if metrics.get('lines', 0) > 3:
            score += 5
            
        return max(0, min(100, score))
        
    def _generate_suggestions(self, 
                             node: ast.AST,
                             docstring: str,
                             issues: List[DocstringIssue]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if not docstring:
            suggestions.append("Add a comprehensive docstring")
        else:
            if len(docstring) < 50:
                suggestions.append("Consider adding more detail to the docstring")
                
            if not any('example' in i.lower() for i in docstring.split('\n')):
                suggestions.append("Add usage examples")
                
            if isinstance(node, ast.ClassDef):
                suggestions.append("Document class attributes and methods")
                
        return suggestions
        
    def generate_report(self, analyses: Dict[str, DocstringAnalysis]) -> str:
        """
        Generate quality report.
        
        Args:
            analyses: Analysis results
            
        Returns:
            Formatted report
        """
        total_score = sum(a.score for a in analyses.values()) / max(len(analyses), 1)
        
        report = [
            "# Docstring Quality Report",
            "",
            f"**Overall Score:** {total_score:.1f}/100",
            f"**Functions/Classes Analyzed:** {len(analyses)}",
            f"**Style:** {self.style.value}",
            "",
            "## Summary",
            ""
        ]
        
        # Count issues by severity
        errors = sum(1 for a in analyses.values() for i in a.issues if i.severity == "error")
        warnings = sum(1 for a in analyses.values() for i in a.issues if i.severity == "warning")
        info = sum(1 for a in analyses.values() for i in a.issues if i.severity == "info")
        
        report.extend([
            f"- Errors: {errors}",
            f"- Warnings: {warnings}",
            f"- Info: {info}",
            "",
            "## Detailed Analysis",
            ""
        ])
        
        for name, analysis in analyses.items():
            report.extend([
                f"### {name}",
                f"Score: {analysis.score:.1f}/100",
                ""
            ])
            
            if analysis.issues:
                report.append("**Issues:**")
                for issue in analysis.issues:
                    report.append(f"- [{issue.severity}] {issue.message}")
                report.append("")
                
            if analysis.suggestions:
                report.append("**Suggestions:**")
                for suggestion in analysis.suggestions:
                    report.append(f"- {suggestion}")
                report.append("")
                
        return "\n".join(report)
        
    def _load_style_patterns(self) -> Dict[str, Any]:
        """Load style-specific patterns."""
        if self.style == DocstringStyle.GOOGLE:
            return {
                'params': r'Args:\s*\n(.*?)(?:\n\n|\Z)',
                'returns': r'Returns:\s*\n(.*?)(?:\n\n|\Z)',
                'raises': r'Raises:\s*\n(.*?)(?:\n\n|\Z)'
            }
        elif self.style == DocstringStyle.NUMPY:
            return {
                'params': r'Parameters\s*\n-+\s*\n(.*?)(?:\n\n|\Z)',
                'returns': r'Returns\s*\n-+\s*\n(.*?)(?:\n\n|\Z)',
                'raises': r'Raises\s*\n-+\s*\n(.*?)(?:\n\n|\Z)'
            }
        return {}