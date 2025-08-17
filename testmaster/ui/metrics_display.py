"""
Coverage and Quality Metrics Display

Inspired by PraisonAI's performance metrics visualization
for displaying coverage heatmaps, quality scores, and trends.

Features:
- Coverage heatmaps and visualizations
- Quality score displays with trends
- Performance metrics and analytics
- Interactive charts and graphs
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics

from ..core.layer_manager import requires_layer


class MetricType(Enum):
    """Types of metrics to display."""
    COVERAGE = "coverage"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    TREND = "trend"


class ChartType(Enum):
    """Types of charts for visualization."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    HEATMAP = "heatmap"
    GAUGE = "gauge"


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None


@dataclass
class CoverageMetrics:
    """Coverage-related metrics."""
    overall_coverage: float
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    files_covered: int
    total_files: int
    uncovered_lines: int
    total_lines: int
    coverage_by_file: Dict[str, float] = field(default_factory=dict)
    coverage_history: List[MetricPoint] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Quality-related metrics."""
    overall_quality_score: float
    test_quality_score: float
    code_complexity_score: float
    maintainability_score: float
    test_completeness: float
    passing_tests: int
    total_tests: int
    failed_tests: int
    flaky_tests: int
    quality_history: List[MetricPoint] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance-related metrics."""
    avg_test_duration: float
    total_test_time: float
    tests_per_minute: float
    queue_utilization: float
    worker_utilization: float
    success_rate: float
    idle_module_count: int
    active_module_count: int
    performance_history: List[MetricPoint] = field(default_factory=list)


class MetricsDisplay:
    """
    Display system for coverage and quality metrics.
    
    Generates visualizations and widgets for the dashboard.
    """
    
    @requires_layer("layer2_monitoring", "metrics_display")
    def __init__(self, max_history_points: int = 100):
        """
        Initialize metrics display.
        
        Args:
            max_history_points: Maximum history points to maintain
        """
        self.max_history_points = max_history_points
        
        # Metric data storage
        self.coverage_metrics = CoverageMetrics(
            overall_coverage=0.0,
            line_coverage=0.0,
            branch_coverage=0.0,
            function_coverage=0.0,
            files_covered=0,
            total_files=0,
            uncovered_lines=0,
            total_lines=0
        )
        
        self.quality_metrics = QualityMetrics(
            overall_quality_score=0.0,
            test_quality_score=0.0,
            code_complexity_score=0.0,
            maintainability_score=0.0,
            test_completeness=0.0,
            passing_tests=0,
            total_tests=0,
            failed_tests=0,
            flaky_tests=0
        )
        
        self.performance_metrics = PerformanceMetrics(
            avg_test_duration=0.0,
            total_test_time=0.0,
            tests_per_minute=0.0,
            queue_utilization=0.0,
            worker_utilization=0.0,
            success_rate=0.0,
            idle_module_count=0,
            active_module_count=0
        )
        
        print("📊 Metrics display initialized")
    
    def update_coverage_metrics(self, coverage_data: Dict[str, Any]):
        """
        Update coverage metrics from coverage data.
        
        Args:
            coverage_data: Coverage analysis results
        """
        # Update current metrics
        self.coverage_metrics.overall_coverage = coverage_data.get('overall_coverage', 0.0)
        self.coverage_metrics.line_coverage = coverage_data.get('line_coverage', 0.0)
        self.coverage_metrics.branch_coverage = coverage_data.get('branch_coverage', 0.0)
        self.coverage_metrics.function_coverage = coverage_data.get('function_coverage', 0.0)
        self.coverage_metrics.files_covered = coverage_data.get('files_covered', 0)
        self.coverage_metrics.total_files = coverage_data.get('total_files', 0)
        self.coverage_metrics.uncovered_lines = coverage_data.get('uncovered_lines', 0)
        self.coverage_metrics.total_lines = coverage_data.get('total_lines', 0)
        
        # Update file-level coverage
        if 'file_coverage' in coverage_data:
            self.coverage_metrics.coverage_by_file = coverage_data['file_coverage']
        
        # Add to history
        history_point = MetricPoint(
            timestamp=datetime.now(),
            value=self.coverage_metrics.overall_coverage,
            metadata={
                'line_coverage': self.coverage_metrics.line_coverage,
                'branch_coverage': self.coverage_metrics.branch_coverage,
                'function_coverage': self.coverage_metrics.function_coverage
            }
        )
        
        self.coverage_metrics.coverage_history.append(history_point)
        self._trim_history(self.coverage_metrics.coverage_history)
    
    def update_quality_metrics(self, quality_data: Dict[str, Any]):
        """
        Update quality metrics from quality analysis.
        
        Args:
            quality_data: Quality analysis results
        """
        # Update current metrics
        self.quality_metrics.overall_quality_score = quality_data.get('overall_quality', 0.0)
        self.quality_metrics.test_quality_score = quality_data.get('test_quality', 0.0)
        self.quality_metrics.code_complexity_score = quality_data.get('complexity_score', 0.0)
        self.quality_metrics.maintainability_score = quality_data.get('maintainability', 0.0)
        self.quality_metrics.test_completeness = quality_data.get('test_completeness', 0.0)
        self.quality_metrics.passing_tests = quality_data.get('passing_tests', 0)
        self.quality_metrics.total_tests = quality_data.get('total_tests', 0)
        self.quality_metrics.failed_tests = quality_data.get('failed_tests', 0)
        self.quality_metrics.flaky_tests = quality_data.get('flaky_tests', 0)
        
        # Add to history
        history_point = MetricPoint(
            timestamp=datetime.now(),
            value=self.quality_metrics.overall_quality_score,
            metadata={
                'test_quality': self.quality_metrics.test_quality_score,
                'complexity': self.quality_metrics.code_complexity_score,
                'maintainability': self.quality_metrics.maintainability_score
            }
        )
        
        self.quality_metrics.quality_history.append(history_point)
        self._trim_history(self.quality_metrics.quality_history)
    
    def update_performance_metrics(self, performance_data: Dict[str, Any]):
        """
        Update performance metrics from performance data.
        
        Args:
            performance_data: Performance analysis results
        """
        # Update current metrics
        self.performance_metrics.avg_test_duration = performance_data.get('avg_test_duration', 0.0)
        self.performance_metrics.total_test_time = performance_data.get('total_test_time', 0.0)
        self.performance_metrics.tests_per_minute = performance_data.get('tests_per_minute', 0.0)
        self.performance_metrics.queue_utilization = performance_data.get('queue_utilization', 0.0)
        self.performance_metrics.worker_utilization = performance_data.get('worker_utilization', 0.0)
        self.performance_metrics.success_rate = performance_data.get('success_rate', 0.0)
        self.performance_metrics.idle_module_count = performance_data.get('idle_modules', 0)
        self.performance_metrics.active_module_count = performance_data.get('active_modules', 0)
        
        # Add to history
        history_point = MetricPoint(
            timestamp=datetime.now(),
            value=self.performance_metrics.success_rate,
            metadata={
                'avg_duration': self.performance_metrics.avg_test_duration,
                'tests_per_minute': self.performance_metrics.tests_per_minute,
                'queue_utilization': self.performance_metrics.queue_utilization
            }
        )
        
        self.performance_metrics.performance_history.append(history_point)
        self._trim_history(self.performance_metrics.performance_history)
    
    def _trim_history(self, history: List[MetricPoint]):
        """Trim history to maximum size."""
        if len(history) > self.max_history_points:
            history[:] = history[-self.max_history_points:]
    
    def get_coverage_widget_data(self) -> Dict[str, Any]:
        """Get data for coverage widget."""
        return {
            'type': 'coverage',
            'overall_coverage': self.coverage_metrics.overall_coverage,
            'line_coverage': self.coverage_metrics.line_coverage,
            'branch_coverage': self.coverage_metrics.branch_coverage,
            'function_coverage': self.coverage_metrics.function_coverage,
            'files_covered': self.coverage_metrics.files_covered,
            'total_files': self.coverage_metrics.total_files,
            'coverage_trend': self._calculate_trend(self.coverage_metrics.coverage_history),
            'chart_data': self._prepare_chart_data(
                self.coverage_metrics.coverage_history, 
                ChartType.LINE
            )
        }
    
    def get_quality_widget_data(self) -> Dict[str, Any]:
        """Get data for quality widget."""
        return {
            'type': 'quality',
            'overall_quality': self.quality_metrics.overall_quality_score,
            'test_quality': self.quality_metrics.test_quality_score,
            'complexity_score': self.quality_metrics.code_complexity_score,
            'maintainability': self.quality_metrics.maintainability_score,
            'test_completeness': self.quality_metrics.test_completeness,
            'passing_tests': self.quality_metrics.passing_tests,
            'total_tests': self.quality_metrics.total_tests,
            'failed_tests': self.quality_metrics.failed_tests,
            'quality_trend': self._calculate_trend(self.quality_metrics.quality_history),
            'chart_data': self._prepare_chart_data(
                self.quality_metrics.quality_history,
                ChartType.GAUGE
            )
        }
    
    def get_performance_widget_data(self) -> Dict[str, Any]:
        """Get data for performance widget."""
        return {
            'type': 'performance',
            'avg_test_duration': self.performance_metrics.avg_test_duration,
            'tests_per_minute': self.performance_metrics.tests_per_minute,
            'success_rate': self.performance_metrics.success_rate,
            'queue_utilization': self.performance_metrics.queue_utilization,
            'worker_utilization': self.performance_metrics.worker_utilization,
            'idle_modules': self.performance_metrics.idle_module_count,
            'active_modules': self.performance_metrics.active_module_count,
            'performance_trend': self._calculate_trend(self.performance_metrics.performance_history),
            'chart_data': self._prepare_chart_data(
                self.performance_metrics.performance_history,
                ChartType.BAR
            )
        }
    
    def get_coverage_heatmap_data(self) -> Dict[str, Any]:
        """Generate coverage heatmap data."""
        heatmap_data = []
        
        for file_path, coverage in self.coverage_metrics.coverage_by_file.items():
            # Create heatmap entry
            file_name = Path(file_path).name
            
            # Determine color based on coverage
            if coverage >= 90:
                color = "#00b894"  # Green
            elif coverage >= 70:
                color = "#fdcb6e"  # Yellow
            elif coverage >= 50:
                color = "#e17055"  # Orange
            else:
                color = "#d63031"  # Red
            
            heatmap_data.append({
                'file': file_name,
                'full_path': file_path,
                'coverage': coverage,
                'color': color,
                'size': max(10, min(100, coverage))  # Size based on coverage
            })
        
        return {
            'type': 'heatmap',
            'data': heatmap_data,
            'chart_data': self._prepare_heatmap_chart_data(heatmap_data)
        }
    
    def get_trend_analysis(self, metric_type: MetricType, hours: int = 24) -> Dict[str, Any]:
        """
        Get trend analysis for a specific metric type.
        
        Args:
            metric_type: Type of metric to analyze
            hours: Number of hours to analyze
            
        Returns:
            Trend analysis data
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        if metric_type == MetricType.COVERAGE:
            history = self.coverage_metrics.coverage_history
            current_value = self.coverage_metrics.overall_coverage
        elif metric_type == MetricType.QUALITY:
            history = self.quality_metrics.quality_history
            current_value = self.quality_metrics.overall_quality_score
        elif metric_type == MetricType.PERFORMANCE:
            history = self.performance_metrics.performance_history
            current_value = self.performance_metrics.success_rate
        else:
            return {'error': 'Unknown metric type'}
        
        # Filter recent history
        recent_history = [
            point for point in history
            if point.timestamp > cutoff
        ]
        
        if len(recent_history) < 2:
            return {
                'trend': 'insufficient_data',
                'change_percent': 0.0,
                'current_value': current_value,
                'data_points': len(recent_history)
            }
        
        # Calculate trend
        values = [point.value for point in recent_history]
        timestamps = [point.timestamp for point in recent_history]
        
        # Simple linear trend calculation
        first_value = values[0]
        last_value = values[-1]
        change_percent = ((last_value - first_value) / max(first_value, 0.1)) * 100
        
        # Determine trend direction
        if abs(change_percent) < 1:
            trend = 'stable'
        elif change_percent > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        # Calculate statistics
        avg_value = statistics.mean(values)
        min_value = min(values)
        max_value = max(values)
        
        return {
            'trend': trend,
            'change_percent': change_percent,
            'current_value': current_value,
            'avg_value': avg_value,
            'min_value': min_value,
            'max_value': max_value,
            'data_points': len(recent_history),
            'time_range_hours': hours,
            'chart_data': self._prepare_trend_chart_data(recent_history)
        }
    
    def _calculate_trend(self, history: List[MetricPoint], window: int = 10) -> str:
        """Calculate trend from metric history."""
        if len(history) < 2:
            return 'stable'
        
        # Use last N points for trend
        recent_points = history[-window:]
        if len(recent_points) < 2:
            return 'stable'
        
        first_value = recent_points[0].value
        last_value = recent_points[-1].value
        
        change_percent = ((last_value - first_value) / max(first_value, 0.1)) * 100
        
        if abs(change_percent) < 2:
            return 'stable'
        elif change_percent > 0:
            return 'improving'
        else:
            return 'declining'
    
    def _prepare_chart_data(self, history: List[MetricPoint], chart_type: ChartType) -> Dict[str, Any]:
        """Prepare chart data for visualization."""
        if not history:
            return {'labels': [], 'values': []}
        
        labels = [point.timestamp.strftime('%H:%M') for point in history[-20:]]  # Last 20 points
        values = [point.value for point in history[-20:]]
        
        if chart_type == ChartType.LINE:
            return {
                'type': 'line',
                'labels': labels,
                'datasets': [{
                    'label': 'Value',
                    'data': values,
                    'borderColor': '#74b9ff',
                    'backgroundColor': 'rgba(116, 185, 255, 0.1)',
                    'tension': 0.4
                }]
            }
        
        elif chart_type == ChartType.BAR:
            return {
                'type': 'bar',
                'labels': labels,
                'datasets': [{
                    'label': 'Value',
                    'data': values,
                    'backgroundColor': '#00b894',
                    'borderColor': '#00a085',
                    'borderWidth': 1
                }]
            }
        
        elif chart_type == ChartType.GAUGE:
            # For gauge charts, return current value and ranges
            current_value = values[-1] if values else 0
            return {
                'type': 'gauge',
                'value': current_value,
                'min': 0,
                'max': 100,
                'ranges': [
                    {'from': 0, 'to': 30, 'color': '#d63031'},
                    {'from': 30, 'to': 70, 'color': '#fdcb6e'},
                    {'from': 70, 'to': 100, 'color': '#00b894'}
                ]
            }
        
        else:
            return {'labels': labels, 'values': values}
    
    def _prepare_heatmap_chart_data(self, heatmap_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare heatmap chart data."""
        return {
            'type': 'heatmap',
            'data': [
                {
                    'x': item['file'],
                    'y': 'Coverage',
                    'v': item['coverage'],
                    'color': item['color']
                }
                for item in heatmap_data
            ]
        }
    
    def _prepare_trend_chart_data(self, history: List[MetricPoint]) -> Dict[str, Any]:
        """Prepare trend chart data."""
        if not history:
            return {'labels': [], 'values': []}
        
        labels = [point.timestamp.strftime('%m/%d %H:%M') for point in history]
        values = [point.value for point in history]
        
        return {
            'type': 'line',
            'labels': labels,
            'datasets': [{
                'label': 'Trend',
                'data': values,
                'borderColor': '#6c5ce7',
                'backgroundColor': 'rgba(108, 92, 231, 0.1)',
                'tension': 0.4,
                'fill': True
            }]
        }
    
    def get_all_widget_data(self) -> Dict[str, Any]:
        """Get all widget data for dashboard."""
        return {
            'coverage': self.get_coverage_widget_data(),
            'quality': self.get_quality_widget_data(),
            'performance': self.get_performance_widget_data(),
            'coverage_heatmap': self.get_coverage_heatmap_data(),
            'coverage_trend': self.get_trend_analysis(MetricType.COVERAGE),
            'quality_trend': self.get_trend_analysis(MetricType.QUALITY),
            'performance_trend': self.get_trend_analysis(MetricType.PERFORMANCE),
            'last_updated': datetime.now().isoformat()
        }
    
    def export_metrics_report(self, output_path: str = "metrics_report.json"):
        """Export comprehensive metrics report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'coverage_metrics': {
                'overall_coverage': self.coverage_metrics.overall_coverage,
                'line_coverage': self.coverage_metrics.line_coverage,
                'branch_coverage': self.coverage_metrics.branch_coverage,
                'function_coverage': self.coverage_metrics.function_coverage,
                'files_covered': self.coverage_metrics.files_covered,
                'total_files': self.coverage_metrics.total_files,
                'coverage_by_file': self.coverage_metrics.coverage_by_file
            },
            'quality_metrics': {
                'overall_quality_score': self.quality_metrics.overall_quality_score,
                'test_quality_score': self.quality_metrics.test_quality_score,
                'code_complexity_score': self.quality_metrics.code_complexity_score,
                'maintainability_score': self.quality_metrics.maintainability_score,
                'test_completeness': self.quality_metrics.test_completeness,
                'passing_tests': self.quality_metrics.passing_tests,
                'total_tests': self.quality_metrics.total_tests,
                'failed_tests': self.quality_metrics.failed_tests,
                'flaky_tests': self.quality_metrics.flaky_tests
            },
            'performance_metrics': {
                'avg_test_duration': self.performance_metrics.avg_test_duration,
                'total_test_time': self.performance_metrics.total_test_time,
                'tests_per_minute': self.performance_metrics.tests_per_minute,
                'queue_utilization': self.performance_metrics.queue_utilization,
                'worker_utilization': self.performance_metrics.worker_utilization,
                'success_rate': self.performance_metrics.success_rate,
                'idle_module_count': self.performance_metrics.idle_module_count,
                'active_module_count': self.performance_metrics.active_module_count
            },
            'trends': {
                'coverage_trend': self.get_trend_analysis(MetricType.COVERAGE),
                'quality_trend': self.get_trend_analysis(MetricType.QUALITY),
                'performance_trend': self.get_trend_analysis(MetricType.PERFORMANCE)
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Metrics report exported to {output_path}")
        except Exception as e:
            print(f"⚠️ Error exporting metrics report: {e}")


# Widget Classes for Specific Displays

class CoverageWidget:
    """Specialized widget for coverage display."""
    
    def __init__(self, metrics_display: MetricsDisplay):
        self.metrics_display = metrics_display
    
    def get_widget_html(self) -> str:
        """Generate HTML for coverage widget."""
        data = self.metrics_display.get_coverage_widget_data()
        
        return f"""
        <div class="coverage-widget">
            <h3>📊 Coverage Overview</h3>
            <div class="coverage-gauge">
                <div class="gauge-value">{data['overall_coverage']:.1f}%</div>
                <div class="gauge-label">Overall Coverage</div>
            </div>
            <div class="coverage-breakdown">
                <div class="metric-row">
                    <span>Line Coverage:</span>
                    <span>{data['line_coverage']:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span>Branch Coverage:</span>
                    <span>{data['branch_coverage']:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span>Function Coverage:</span>
                    <span>{data['function_coverage']:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span>Files Covered:</span>
                    <span>{data['files_covered']}/{data['total_files']}</span>
                </div>
            </div>
            <div class="trend-indicator trend-{data['coverage_trend']}">
                Trend: {data['coverage_trend'].title()}
            </div>
        </div>
        """


class QualityWidget:
    """Specialized widget for quality display."""
    
    def __init__(self, metrics_display: MetricsDisplay):
        self.metrics_display = metrics_display
    
    def get_widget_html(self) -> str:
        """Generate HTML for quality widget."""
        data = self.metrics_display.get_quality_widget_data()
        
        return f"""
        <div class="quality-widget">
            <h3>🎯 Quality Metrics</h3>
            <div class="quality-score">
                <div class="score-value">{data['overall_quality']:.0f}</div>
                <div class="score-label">Quality Score</div>
            </div>
            <div class="quality-breakdown">
                <div class="metric-row">
                    <span>Test Quality:</span>
                    <span>{data['test_quality']:.0f}/100</span>
                </div>
                <div class="metric-row">
                    <span>Complexity:</span>
                    <span>{data['complexity_score']:.0f}/100</span>
                </div>
                <div class="metric-row">
                    <span>Maintainability:</span>
                    <span>{data['maintainability']:.0f}/100</span>
                </div>
                <div class="metric-row">
                    <span>Test Results:</span>
                    <span>{data['passing_tests']}/{data['total_tests']} passed</span>
                </div>
            </div>
            <div class="trend-indicator trend-{data['quality_trend']}">
                Trend: {data['quality_trend'].title()}
            </div>
        </div>
        """


# Convenience function for metrics setup
def setup_metrics_display() -> MetricsDisplay:
    """Setup metrics display with default configuration."""
    return MetricsDisplay(max_history_points=100)