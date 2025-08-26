"""
Modularized Technical Debt Analyzer
===================================

Main orchestrator for technical debt analysis using modular components.
Replaces the original 1546-line debt_analyzer.py.
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .debt_base import (
    DebtItem, DebtMetrics, DebtCategory,
    DebtConfiguration, RemediationPlan
)
from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.core.intelligence.debt_code_analyzer import CodeDebtAnalyzer
from .debt_test_analyzer import TestDebtAnalyzer
from .debt_quantifier import DebtQuantifier


class TechnicalDebtAnalyzer:
    """
    Orchestrates technical debt analysis across all components.
    
    This modularized version maintains all functionality of the original
    while being split into focused, maintainable components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the technical debt analyzer."""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize components
        self.code_analyzer = CodeDebtAnalyzer(config)
        self.test_analyzer = TestDebtAnalyzer(config)
        self.quantifier = DebtQuantifier(config)
        
        # Additional analyzers can be added here:
        # self.doc_analyzer = DocumentationDebtAnalyzer(config)
        # self.security_analyzer = SecurityDebtAnalyzer(config)
        # self.infra_analyzer = InfrastructureDebtAnalyzer(config)
        
        # Results storage
        self.debt_items = []
        self.debt_metrics = None
        self.remediation_plan = []
        
        # Configuration
        self.root_path = None
        self.exclude_patterns = self.config.get('exclude_patterns', [
            '__pycache__', '.git', '.pytest_cache', 'venv', 'env',
            'build', 'dist', '*.egg-info'
        ])
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the analyzer."""
        logger = logging.getLogger('debt_analyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def analyze(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive technical debt analysis.
        
        Args:
            root_path: Root directory to analyze (defaults to current)
            
        Returns:
            Comprehensive debt analysis report
        """
        self.root_path = Path(root_path or os.getcwd())
        self.logger.info(f"Starting debt analysis for: {self.root_path}")
        
        # Reset results
        self.debt_items = []
        
        # Get Python files
        source_files = self._get_python_files()
        test_files = self._get_test_files(source_files)
        
        self.logger.info(f"Found {len(source_files)} source files and {len(test_files)} test files")
        
        # Run analyses
        self._analyze_code_debt(source_files)
        self._analyze_test_debt(source_files, test_files)
        # Additional analyses can be added here
        
        # Quantify debt
        self.debt_metrics = self.quantifier.quantify_debt(self.debt_items)
        
        # Create remediation plan
        self.remediation_plan = self.quantifier.prioritize_debt(self.debt_items)
        
        # Track trends
        self.quantifier.track_trend(self.debt_items, self.debt_metrics)
        
        # Generate report
        report = self._generate_report()
        
        self.logger.info(f"Analysis complete: {len(self.debt_items)} debt items found")
        self.logger.info(f"Total debt: {self.debt_metrics.total_debt_hours:.1f} hours")
        
        return report
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project."""
        python_files = []
        
        for pattern in ['**/*.py']:
            for file_path in self.root_path.glob(pattern):
                # Skip excluded paths
                if not self._should_exclude(file_path):
                    python_files.append(file_path)
        
        return python_files
    
    def _get_test_files(self, all_files: List[Path]) -> List[Path]:
        """Identify test files from all Python files."""
        test_patterns = ['test_', '_test', 'Test', 'spec_', '_spec']
        test_files = []
        
        for file_path in all_files:
            file_name = file_path.stem
            if any(pattern in file_name for pattern in test_patterns):
                test_files.append(file_path)
            elif 'test' in file_path.parts or 'tests' in file_path.parts:
                test_files.append(file_path)
        
        return test_files
    
    def _should_exclude(self, file_path: Path) -> bool:
        """Check if path should be excluded from analysis."""
        path_str = str(file_path)
        
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return True
        
        return False
    
    def _analyze_code_debt(self, source_files: List[Path]):
        """Analyze code quality debt."""
        self.logger.info("Analyzing code quality debt...")
        
        code_debt_items = self.code_analyzer.analyze_code_debt(source_files)
        self.debt_items.extend(code_debt_items)
        
        self.logger.info(f"Found {len(code_debt_items)} code quality issues")
    
    def _analyze_test_debt(self, source_files: List[Path], test_files: List[Path]):
        """Analyze testing debt."""
        self.logger.info("Analyzing test debt...")
        
        # Get coverage data if available
        coverage_data = self._get_coverage_data()
        
        test_debt_items = self.test_analyzer.analyze_test_debt(
            source_files, test_files, coverage_data
        )
        self.debt_items.extend(test_debt_items)
        
        self.logger.info(f"Found {len(test_debt_items)} testing issues")
    
    def _get_coverage_data(self) -> Optional[Dict[str, float]]:
        """Get test coverage data if available."""
        # This would integrate with coverage.py or similar tools
        # For now, returning mock data
        if self.config.get('use_mock_coverage', False):
            return {
                'total_coverage': 65.0,
                'module_a.py': 80.0,
                'module_b.py': 45.0,
                'module_c.py': 70.0
            }
        return None
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive debt report."""
        # Get summary from quantifier
        summary = self.quantifier.generate_summary(
            self.debt_items, self.debt_metrics
        )
        
        # Build full report
        report = {
            'summary': summary,
            'metrics': self.debt_metrics.to_dict() if self.debt_metrics else {},
            'debt_items': [item.to_dict() for item in self.debt_items],
            'remediation_plan': [plan.to_dict() for plan in self.remediation_plan[:10]],  # Top 10
            'trends': [trend.to_dict() for trend in self.quantifier.debt_trends],
            'analysis_metadata': {
                'root_path': str(self.root_path),
                'files_analyzed': len(self._get_python_files()),
                'analyzer_version': '2.0.0',
                'modularized': True
            }
        }
        
        return report
    
    def get_critical_items(self) -> List[DebtItem]:
        """Get critical debt items requiring immediate attention."""
        return [
            item for item in self.debt_items 
            if item.severity in ['critical', 'high']
        ]
    
    def get_quick_wins(self, max_hours: float = 4.0) -> List[DebtItem]:
        """Get debt items that can be fixed quickly."""
        return [
            item for item in self.debt_items
            if item.estimated_hours <= max_hours
        ]
    
    def get_debt_by_category(self) -> Dict[str, List[DebtItem]]:
        """Group debt items by category."""
        categorized = {}
        
        for item in self.debt_items:
            category = (item.category or DebtCategory.CODE_QUALITY).value
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)
        
        return categorized
    
    def export_report(self, format: str = 'json') -> str:
        """Export debt report in specified format."""
        report = self._generate_report()
        
        if format == 'json':
            import json
            return json.dumps(report, indent=2, default=str)
        elif format == 'markdown':
            return self._format_as_markdown(report)
        else:
            return str(report)
    
    def _format_as_markdown(self, report: Dict[str, Any]) -> str:
        """Format report as markdown."""
        summary = report['summary']
        
        md = f"""# Technical Debt Analysis Report

## Summary
- **Total Debt Items**: {summary['total_debt_items']}
- **Total Debt Hours**: {summary['total_debt_hours']:.1f}
- **Debt Ratio**: {summary['debt_ratio_percent']:.1f}%
- **Monthly Interest**: {summary['monthly_interest_hours']:.1f} hours
- **Break-even Point**: {summary['break_even_months']} months
- **Velocity Impact**: {summary['velocity_impact_percent']:.1f}%

## Financial Impact
- **Direct Cost**: ${summary['financial_impact']['direct_cost']:,.0f}
- **Annual Interest Cost**: ${summary['financial_impact']['annual_interest_cost']:,.0f}
- **Annual Productivity Loss**: ${summary['financial_impact']['annual_productivity_loss']:,.0f}

## Top Priority Items
"""
        
        for i, plan in enumerate(report['remediation_plan'][:5], 1):
            item = plan['debt_item']
            md += f"\n{i}. **{item['type']}** ({item['severity']})\n"
            md += f"   - Location: {item['location']}\n"
            md += f"   - Effort: {item['estimated_hours']:.1f} hours\n"
            md += f"   - ROI: {plan['expected_roi']:.1%}\n"
        
        return md


# Export
__all__ = ['TechnicalDebtAnalyzer']