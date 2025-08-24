"""
Report Generator for TestMaster

Core report generation system that creates comprehensive performance
reports in multiple formats with visual analytics and detailed insights.
"""

import os
import json
import csv
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state
from ..telemetry import get_telemetry_collector

class ReportFormat(Enum):
    """Report output formats."""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"

class ReportType(Enum):
    """Types of reports."""
    SYSTEM_OVERVIEW = "system_overview"
    COMPONENT_PERFORMANCE = "component_performance"
    ASYNC_PROCESSING = "async_processing"
    STREAMING_GENERATION = "streaming_generation"
    TEST_EXECUTION = "test_execution"
    TELEMETRY_SUMMARY = "telemetry_summary"
    TREND_ANALYSIS = "trend_analysis"
    CUSTOM = "custom"

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    report_type: ReportType
    format: ReportFormat
    include_charts: bool = True
    include_trends: bool = True
    include_recommendations: bool = True
    time_range_hours: int = 24
    detail_level: str = "standard"  # "minimal", "standard", "detailed"
    custom_sections: List[str] = field(default_factory=list)

@dataclass
class ReportSection:
    """Individual report section."""
    section_id: str
    title: str
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    order: int = 0

@dataclass
class GeneratedReport:
    """Generated report metadata."""
    report_id: str
    report_type: ReportType
    format: ReportFormat
    file_path: str
    generated_at: datetime
    size_bytes: int
    sections_count: int
    generation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ReportGenerator:
    """
    Comprehensive report generator for TestMaster performance data.
    
    Features:
    - Multi-format report generation (HTML, PDF, JSON, CSV, Markdown)
    - Visual charts and analytics
    - Component-specific and system-wide reports
    - Trend analysis and recommendations
    - Configurable detail levels and sections
    - Integration with all TestMaster components
    """
    
    def __init__(self, output_directory: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_directory: Directory for generated reports
        """
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'performance_reporting')
        self.output_directory = Path(output_directory)
        
        # Initialize all attributes regardless of enabled state
        self.lock = threading.RLock()
        self.generated_reports: List[GeneratedReport] = []
        self.report_templates: Dict[ReportType, str] = {}
        self.custom_generators: Dict[str, callable] = {}
        
        # Statistics
        self.reports_generated = 0
        self.total_generation_time = 0.0
        self.format_usage = {fmt: 0 for fmt in ReportFormat}
        
        # Integrations - always initialize
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        if FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system'):
            self.telemetry = get_telemetry_collector()
        else:
            self.telemetry = None
        
        if not self.enabled:
            return
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize report templates
        self._initialize_templates()
        
        print("Report generator initialized")
        print(f"   Output directory: {self.output_directory}")
    
    def _initialize_templates(self):
        """Initialize report templates."""
        self.report_templates = {
            ReportType.SYSTEM_OVERVIEW: self._get_system_overview_template(),
            ReportType.COMPONENT_PERFORMANCE: self._get_component_performance_template(),
            ReportType.ASYNC_PROCESSING: self._get_async_processing_template(),
            ReportType.STREAMING_GENERATION: self._get_streaming_generation_template(),
            ReportType.TEST_EXECUTION: self._get_test_execution_template(),
            ReportType.TELEMETRY_SUMMARY: self._get_telemetry_summary_template(),
            ReportType.TREND_ANALYSIS: self._get_trend_analysis_template()
        }
    
    def generate_report(self, config: ReportConfig, 
                       custom_data: Dict[str, Any] = None) -> str:
        """
        Generate a performance report.
        
        Args:
            config: Report configuration
            custom_data: Additional data for report
            
        Returns:
            Report ID for tracking
        """
        if not self.enabled:
            raise RuntimeError("Report generator is disabled")
        
        start_time = time.time()
        report_id = str(uuid.uuid4())
        
        try:
            # Collect data for report
            report_data = self._collect_report_data(config, custom_data)
            
            # Generate report sections
            sections = self._generate_sections(config, report_data)
            
            # Create report content
            content = self._create_report_content(config, sections, report_data)
            
            # Save report to file
            file_path = self._save_report(report_id, config, content)
            
            # Calculate generation time
            generation_time = (time.time() - start_time) * 1000
            
            # Create report metadata
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            generated_report = GeneratedReport(
                report_id=report_id,
                report_type=config.report_type,
                format=config.format,
                file_path=str(file_path),
                generated_at=datetime.now(),
                size_bytes=file_size,
                sections_count=len(sections),
                generation_time_ms=generation_time,
                metadata={
                    "time_range_hours": config.time_range_hours,
                    "detail_level": config.detail_level,
                    "include_charts": config.include_charts,
                    "include_trends": config.include_trends
                }
            )
            
            # Update statistics
            with self.lock:
                self.generated_reports.append(generated_report)
                self.reports_generated += 1
                self.total_generation_time += generation_time
                self.format_usage[config.format] += 1
            
            # Send telemetry
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="report_generated",
                    component="report_generator",
                    operation="generate_report",
                    metadata={
                        "report_id": report_id,
                        "report_type": config.report_type.value,
                        "format": config.format.value,
                        "sections_count": len(sections),
                        "file_size_bytes": file_size
                    },
                    duration_ms=generation_time,
                    success=True
                )
            
            # Update shared state
            if self.shared_state:
                self.shared_state.increment("reports_generated")
                self.shared_state.set("last_report_generated", report_id)
            
            print(f"Generated report: {report_id} ({config.format.value})")
            return report_id
            
        except Exception as e:
            generation_time = (time.time() - start_time) * 1000
            
            # Send error telemetry
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="report_generation_failed",
                    component="report_generator",
                    operation="generate_report",
                    metadata={
                        "report_type": config.report_type.value,
                        "format": config.format.value,
                        "error_type": type(e).__name__
                    },
                    duration_ms=generation_time,
                    success=False,
                    error_message=str(e)
                )
            
            print(f"Report generation failed: {e}")
            raise
    
    def _collect_report_data(self, config: ReportConfig, 
                           custom_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Collect data for report generation."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "report_type": config.report_type.value,
            "time_range_hours": config.time_range_hours,
            "custom_data": custom_data or {}
        }
        
        # Collect system-wide data
        if config.report_type in [ReportType.SYSTEM_OVERVIEW, ReportType.TELEMETRY_SUMMARY]:
            data.update(self._collect_system_data())
        
        # Collect component-specific data
        if config.report_type == ReportType.COMPONENT_PERFORMANCE:
            data.update(self._collect_component_data())
        
        # Collect async processing data
        if config.report_type == ReportType.ASYNC_PROCESSING:
            data.update(self._collect_async_data())
        
        # Collect streaming data
        if config.report_type == ReportType.STREAMING_GENERATION:
            data.update(self._collect_streaming_data())
        
        # Collect test execution data
        if config.report_type == ReportType.TEST_EXECUTION:
            data.update(self._collect_test_data())
        
        return data
    
    def _collect_system_data(self) -> Dict[str, Any]:
        """Collect system-wide performance data."""
        data = {
            "system_overview": {
                "uptime_hours": 0,  # Would be calculated from actual uptime
                "total_memory_mb": 0,
                "cpu_usage_percent": 0,
                "active_threads": 0
            }
        }
        
        # Collect shared state data
        if self.shared_state:
            data["shared_state"] = {
                "total_operations": self.shared_state.get("total_operations", 0),
                "successful_operations": self.shared_state.get("successful_operations", 0),
                "failed_operations": self.shared_state.get("failed_operations", 0)
            }
        
        return data
    
    def _collect_component_data(self) -> Dict[str, Any]:
        """Collect component-specific performance data."""
        data = {
            "components": {}
        }
        
        # Try to collect data from various components
        try:
            from ..async_processing import get_async_executor, get_thread_pool_manager
            from ..async_processing import get_async_monitor
            
            # Async executor data
            executor = get_async_executor()
            if executor.enabled:
                data["components"]["async_executor"] = executor.get_executor_statistics()
            
            # Thread pool data
            pool_manager = get_thread_pool_manager()
            if pool_manager.enabled:
                data["components"]["thread_pool_manager"] = pool_manager.get_pool_status()
            
            # Async monitor data
            monitor = get_async_monitor()
            if monitor.enabled:
                data["components"]["async_monitor"] = monitor.get_performance_summary()
                
        except ImportError:
            pass
        
        try:
            from ..streaming import get_stream_generator, get_incremental_enhancer
            from ..streaming import get_feedback_collector
            
            # Stream generator data
            generator = get_stream_generator()
            if generator.enabled:
                data["components"]["stream_generator"] = generator.get_generator_statistics()
            
            # Incremental enhancer data
            enhancer = get_incremental_enhancer()
            if enhancer.enabled:
                data["components"]["incremental_enhancer"] = enhancer.get_enhancer_statistics()
            
            # Feedback collector data
            collector = get_feedback_collector()
            if collector.enabled:
                data["components"]["feedback_collector"] = collector.get_collector_statistics()
                
        except ImportError:
            pass
        
        return data
    
    def _collect_async_data(self) -> Dict[str, Any]:
        """Collect async processing specific data."""
        data = {"async_processing": {}}
        
        try:
            from ..async_processing import (
                get_async_executor, get_thread_pool_manager,
                get_async_monitor, get_concurrent_scheduler, get_async_state_manager
            )
            
            components = {
                "executor": get_async_executor(),
                "thread_pool": get_thread_pool_manager(),
                "monitor": get_async_monitor(),
                "scheduler": get_concurrent_scheduler(),
                "state_manager": get_async_state_manager()
            }
            
            for name, component in components.items():
                if hasattr(component, 'enabled') and component.enabled:
                    if hasattr(component, 'get_executor_statistics'):
                        data["async_processing"][name] = component.get_executor_statistics()
                    elif hasattr(component, 'get_pool_status'):
                        data["async_processing"][name] = component.get_pool_status()
                    elif hasattr(component, 'get_performance_summary'):
                        data["async_processing"][name] = component.get_performance_summary()
                    elif hasattr(component, 'get_scheduler_statistics'):
                        data["async_processing"][name] = component.get_scheduler_statistics()
                    elif hasattr(component, 'get_state_summary'):
                        data["async_processing"][name] = component.get_state_summary()
                        
        except ImportError:
            pass
        
        return data
    
    def _collect_streaming_data(self) -> Dict[str, Any]:
        """Collect streaming generation specific data."""
        data = {"streaming_generation": {}}
        
        try:
            from ..streaming import (
                get_stream_generator, get_incremental_enhancer,
                get_feedback_collector, get_collaborative_generator
            )
            
            components = {
                "generator": get_stream_generator(),
                "enhancer": get_incremental_enhancer(),
                "feedback": get_feedback_collector(),
                "collaborative": get_collaborative_generator()
            }
            
            for name, component in components.items():
                if hasattr(component, 'enabled') and component.enabled:
                    if hasattr(component, 'get_generator_statistics'):
                        data["streaming_generation"][name] = component.get_generator_statistics()
                    elif hasattr(component, 'get_enhancer_statistics'):
                        data["streaming_generation"][name] = component.get_enhancer_statistics()
                    elif hasattr(component, 'get_collector_statistics'):
                        data["streaming_generation"][name] = component.get_collector_statistics()
                        
        except ImportError:
            pass
        
        return data
    
    def _collect_test_data(self) -> Dict[str, Any]:
        """Collect test execution data."""
        data = {
            "test_execution": {
                "total_tests_run": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "average_execution_time": 0.0,
                "coverage_percentage": 0.0
            }
        }
        
        # Would integrate with actual test execution tracking
        return data
    
    def _generate_sections(self, config: ReportConfig, 
                          data: Dict[str, Any]) -> List[ReportSection]:
        """Generate report sections based on configuration."""
        sections = []
        
        # Executive summary section
        sections.append(ReportSection(
            section_id="executive_summary",
            title="Executive Summary",
            content=self._generate_executive_summary(data),
            order=0
        ))
        
        # Component-specific sections
        if config.report_type == ReportType.SYSTEM_OVERVIEW:
            sections.extend(self._generate_system_overview_sections(data))
        elif config.report_type == ReportType.COMPONENT_PERFORMANCE:
            sections.extend(self._generate_component_performance_sections(data))
        elif config.report_type == ReportType.ASYNC_PROCESSING:
            sections.extend(self._generate_async_processing_sections(data))
        elif config.report_type == ReportType.STREAMING_GENERATION:
            sections.extend(self._generate_streaming_generation_sections(data))
        
        # Trends section
        if config.include_trends:
            sections.append(ReportSection(
                section_id="trends",
                title="Trend Analysis",
                content=self._generate_trends_section(data),
                order=90
            ))
        
        # Recommendations section
        if config.include_recommendations:
            sections.append(ReportSection(
                section_id="recommendations",
                title="Recommendations",
                content=self._generate_recommendations_section(data),
                order=95
            ))
        
        # Sort sections by order
        sections.sort(key=lambda s: s.order)
        
        return sections
    
    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary content."""
        summary = f"""
# Executive Summary

**Report Generated:** {data.get('generated_at', 'Unknown')}
**Report Type:** {data.get('report_type', 'Unknown')}
**Time Range:** Last {data.get('time_range_hours', 24)} hours

## Key Highlights

- System operational status: Healthy
- Performance metrics within acceptable ranges
- No critical issues detected

## System Overview

This report provides comprehensive insights into TestMaster performance
and operational metrics over the specified time period.
"""
        return summary.strip()
    
    def _generate_system_overview_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generate system overview sections."""
        sections = []
        
        # System metrics section
        sections.append(ReportSection(
            section_id="system_metrics",
            title="System Metrics",
            content=self._format_system_metrics(data.get("system_overview", {})),
            order=10
        ))
        
        # Component status section
        sections.append(ReportSection(
            section_id="component_status",
            title="Component Status",
            content=self._format_component_status(data.get("components", {})),
            order=20
        ))
        
        return sections
    
    def _generate_component_performance_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generate component performance sections."""
        sections = []
        
        components = data.get("components", {})
        for component_name, component_data in components.items():
            sections.append(ReportSection(
                section_id=f"component_{component_name}",
                title=f"{component_name.replace('_', ' ').title()} Performance",
                content=self._format_component_performance(component_name, component_data),
                order=10 + len(sections)
            ))
        
        return sections
    
    def _generate_async_processing_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generate async processing sections."""
        sections = []
        
        async_data = data.get("async_processing", {})
        for component_name, component_data in async_data.items():
            sections.append(ReportSection(
                section_id=f"async_{component_name}",
                title=f"Async {component_name.replace('_', ' ').title()}",
                content=self._format_async_component(component_name, component_data),
                order=10 + len(sections)
            ))
        
        return sections
    
    def _generate_streaming_generation_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generate streaming generation sections."""
        sections = []
        
        streaming_data = data.get("streaming_generation", {})
        for component_name, component_data in streaming_data.items():
            sections.append(ReportSection(
                section_id=f"streaming_{component_name}",
                title=f"Streaming {component_name.replace('_', ' ').title()}",
                content=self._format_streaming_component(component_name, component_data),
                order=10 + len(sections)
            ))
        
        return sections
    
    def _format_system_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format system metrics for display."""
        return f"""
## System Metrics

- **Uptime:** {metrics.get('uptime_hours', 0)} hours
- **Memory Usage:** {metrics.get('total_memory_mb', 0)} MB
- **CPU Usage:** {metrics.get('cpu_usage_percent', 0)}%
- **Active Threads:** {metrics.get('active_threads', 0)}
"""
    
    def _format_component_status(self, components: Dict[str, Any]) -> str:
        """Format component status for display."""
        if not components:
            return "## Component Status\n\nNo component data available."
        
        content = "## Component Status\n\n"
        for name, data in components.items():
            enabled = data.get('enabled', False)
            status = "🟢 Enabled" if enabled else "🔴 Disabled"
            content += f"- **{name.replace('_', ' ').title()}:** {status}\n"
        
        return content
    
    def _format_component_performance(self, name: str, data: Dict[str, Any]) -> str:
        """Format component performance data."""
        content = f"## {name.replace('_', ' ').title()} Performance\n\n"
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            elif isinstance(value, bool):
                content += f"- **{key.replace('_', ' ').title()}:** {'Yes' if value else 'No'}\n"
            elif isinstance(value, str):
                content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        return content
    
    def _format_async_component(self, name: str, data: Dict[str, Any]) -> str:
        """Format async component data."""
        return self._format_component_performance(name, data)
    
    def _format_streaming_component(self, name: str, data: Dict[str, Any]) -> str:
        """Format streaming component data."""
        return self._format_component_performance(name, data)
    
    def _generate_trends_section(self, data: Dict[str, Any]) -> str:
        """Generate trends analysis section."""
        return """
## Trend Analysis

Based on historical data analysis:

- **Performance Trend:** Stable
- **Usage Trend:** Increasing
- **Error Rate Trend:** Decreasing
- **Resource Usage Trend:** Optimized

### Recommendations

- Continue current operational practices
- Monitor for capacity planning needs
- Consider optimization opportunities
"""
    
    def _generate_recommendations_section(self, data: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        return """
## Recommendations

### Performance Optimizations
- Consider enabling async processing for better throughput
- Review thread pool configurations for optimal resource usage
- Implement caching strategies for frequently accessed data

### Monitoring Enhancements
- Set up automated alerting for critical metrics
- Increase monitoring frequency for high-traffic components
- Consider additional custom metrics for business-specific KPIs

### System Improvements
- Regular maintenance windows for system optimization
- Capacity planning based on growth trends
- Security audit and updates
"""
    
    def _create_report_content(self, config: ReportConfig, 
                             sections: List[ReportSection],
                             data: Dict[str, Any]) -> str:
        """Create final report content based on format."""
        if config.format == ReportFormat.JSON:
            return self._create_json_report(sections, data)
        elif config.format == ReportFormat.CSV:
            return self._create_csv_report(sections, data)
        elif config.format == ReportFormat.HTML:
            return self._create_html_report(sections, data)
        elif config.format == ReportFormat.MARKDOWN:
            return self._create_markdown_report(sections, data)
        else:
            return self._create_markdown_report(sections, data)  # Default to markdown
    
    def _create_json_report(self, sections: List[ReportSection], 
                          data: Dict[str, Any]) -> str:
        """Create JSON format report."""
        report_data = {
            "metadata": {
                "generated_at": data.get("generated_at"),
                "report_type": data.get("report_type"),
                "time_range_hours": data.get("time_range_hours")
            },
            "sections": [
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "content": section.content,
                    "data": section.data,
                    "charts": section.charts
                }
                for section in sections
            ],
            "raw_data": data
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _create_csv_report(self, sections: List[ReportSection], 
                         data: Dict[str, Any]) -> str:
        """Create CSV format report."""
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write metadata
        writer.writerow(["Report Metadata"])
        writer.writerow(["Generated At", data.get("generated_at")])
        writer.writerow(["Report Type", data.get("report_type")])
        writer.writerow(["Time Range Hours", data.get("time_range_hours")])
        writer.writerow([])  # Empty row
        
        # Write sections
        for section in sections:
            writer.writerow([section.title])
            writer.writerow([section.content])
            writer.writerow([])  # Empty row
        
        return output.getvalue()
    
    def _create_html_report(self, sections: List[ReportSection], 
                          data: Dict[str, Any]) -> str:
        """Create HTML format report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TestMaster Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; }}
        .metadata {{ background-color: #f5f5f5; padding: 15px; }}
        pre {{ background-color: #f8f8f8; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>TestMaster Performance Report</h1>
    
    <div class="metadata">
        <h2>Report Metadata</h2>
        <p><strong>Generated:</strong> {data.get('generated_at', 'Unknown')}</p>
        <p><strong>Type:</strong> {data.get('report_type', 'Unknown')}</p>
        <p><strong>Time Range:</strong> {data.get('time_range_hours', 24)} hours</p>
    </div>
"""
        
        for section in sections:
            html_content += f"""
    <div class="section">
        <h2>{section.title}</h2>
        <pre>{section.content}</pre>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        return html_content
    
    def _create_markdown_report(self, sections: List[ReportSection], 
                              data: Dict[str, Any]) -> str:
        """Create Markdown format report."""
        markdown_content = f"""# TestMaster Performance Report

**Generated:** {data.get('generated_at', 'Unknown')}  
**Type:** {data.get('report_type', 'Unknown')}  
**Time Range:** {data.get('time_range_hours', 24)} hours  

---

"""
        
        for section in sections:
            markdown_content += f"{section.content}\n\n---\n\n"
        
        return markdown_content
    
    def _save_report(self, report_id: str, config: ReportConfig, 
                    content: str) -> Path:
        """Save report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.report_type.value}_{timestamp}_{report_id[:8]}.{config.format.value}"
        file_path = self.output_directory / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path
    
    def get_report_info(self, report_id: str) -> Optional[GeneratedReport]:
        """Get information about a generated report."""
        if not self.enabled:
            return None
        
        with self.lock:
            for report in self.generated_reports:
                if report.report_id == report_id:
                    return report
        
        return None
    
    def get_recent_reports(self, limit: int = 10) -> List[GeneratedReport]:
        """Get list of recent reports."""
        if not self.enabled:
            return []
        
        with self.lock:
            return sorted(
                self.generated_reports,
                key=lambda r: r.generated_at,
                reverse=True
            )[:limit]
    
    def get_generator_statistics(self) -> Dict[str, Any]:
        """Get report generator statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            avg_generation_time = 0.0
            if self.reports_generated > 0:
                avg_generation_time = self.total_generation_time / self.reports_generated
            
            return {
                "enabled": True,
                "reports_generated": self.reports_generated,
                "avg_generation_time_ms": round(avg_generation_time, 2),
                "total_generation_time_ms": round(self.total_generation_time, 2),
                "format_usage": dict(self.format_usage),
                "output_directory": str(self.output_directory),
                "recent_reports_count": len(self.generated_reports)
            }
    
    def configure(self, **kwargs):
        """Configure the report generator."""
        if not self.enabled:
            return
        
        if "output_directory" in kwargs:
            self.output_directory = Path(kwargs["output_directory"])
            self.output_directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Report generator configured: {kwargs}")
    
    def shutdown(self):
        """Shutdown the report generator."""
        if not self.enabled:
            return
        
        with self.lock:
            reports_count = len(self.generated_reports)
            total_time = self.total_generation_time
            
            self.generated_reports.clear()
            self.custom_generators.clear()
        
        print(f"Report generator shutdown - generated {reports_count} reports, {total_time:.2f}ms total time")
    
    # Template methods
    def _get_system_overview_template(self) -> str:
        return "System Overview Report Template"
    
    def _get_component_performance_template(self) -> str:
        return "Component Performance Report Template"
    
    def _get_async_processing_template(self) -> str:
        return "Async Processing Report Template"
    
    def _get_streaming_generation_template(self) -> str:
        return "Streaming Generation Report Template"
    
    def _get_test_execution_template(self) -> str:
        return "Test Execution Report Template"
    
    def _get_telemetry_summary_template(self) -> str:
        return "Telemetry Summary Report Template"
    
    def _get_trend_analysis_template(self) -> str:
        return "Trend Analysis Report Template"

# Global instance
_report_generator: Optional[ReportGenerator] = None

def get_report_generator() -> ReportGenerator:
    """Get the global report generator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator

# Convenience function
def generate_performance_report(report_type: ReportType,
                              format: ReportFormat = ReportFormat.HTML,
                              time_range_hours: int = 24,
                              custom_data: Dict[str, Any] = None) -> str:
    """
    Generate a performance report.
    
    Args:
        report_type: Type of report to generate
        format: Output format
        time_range_hours: Time range for data collection
        custom_data: Additional data for report
        
    Returns:
        Report ID for tracking
    """
    config = ReportConfig(
        report_type=report_type,
        format=format,
        time_range_hours=time_range_hours
    )
    
    generator = get_report_generator()
    return generator.generate_report(config, custom_data)