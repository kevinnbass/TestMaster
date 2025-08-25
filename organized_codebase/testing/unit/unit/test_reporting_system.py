"""
Test script for TestMaster Performance Report Generator

Comprehensive testing of reporting system components:
- ReportGenerator: Multi-format report generation with visual analytics
- DataCollector: Performance data collection from all components
- DashboardBuilder: Interactive dashboard creation
- MetricsAnalyzer: Trend analysis and anomaly detection
- ReportScheduler: Automated report generation and distribution
"""

import asyncio
import time
import os
from datetime import datetime
from pathlib import Path
from testmaster.core.feature_flags import FeatureFlags
from testmaster.reporting import (
    # Core components
    ReportGenerator, DataCollector, DashboardBuilder,
    MetricsAnalyzer, ReportScheduler,
    
    # Convenience functions
    generate_performance_report, collect_system_metrics,
    build_performance_dashboard, analyze_performance_metrics,
    schedule_automated_reports,
    
    # Enums and configs
    ReportFormat, ReportType, ReportConfig, ChartType,
    DataSource, AnalysisType, DeliveryMethod,
    
    # Global instances
    get_report_generator, get_data_collector, get_dashboard_builder,
    get_metrics_analyzer, get_report_scheduler,
    
    # Utilities
    is_reporting_enabled, configure_reporting, shutdown_reporting
)

class ReportingSystemTest:
    """Comprehensive test suite for reporting system."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.test_output_dir = Path("test_reports")
        
    async def run_all_tests(self):
        """Run all reporting system tests."""
        print("=" * 60)
        print("TestMaster Performance Report Generator Test")
        print("=" * 60)
        
        # Initialize feature flags
        FeatureFlags.initialize("testmaster_config.yaml")
        
        # Check if reporting is enabled
        if not is_reporting_enabled():
            print("[!] Performance reporting is disabled")
            return
        
        print("[+] Performance reporting is enabled")
        
        # Configure reporting
        configure_reporting(
            output_dir=str(self.test_output_dir),
            enable_scheduling=True,
            enable_analytics=True
        )
        
        # Test individual components
        await self.test_report_generator()
        await self.test_data_collector()
        await self.test_dashboard_builder()
        await self.test_metrics_analyzer()
        await self.test_report_scheduler()
        await self.test_integration()
        
        # Display results
        self.display_results()
    
    async def test_report_generator(self):
        """Test ReportGenerator functionality."""
        print("\\n[*] Testing ReportGenerator...")
        
        try:
            generator = get_report_generator()
            
            # Test different report types and formats
            test_configs = [
                (ReportType.SYSTEM_OVERVIEW, ReportFormat.HTML),
                (ReportType.COMPONENT_PERFORMANCE, ReportFormat.JSON),
                (ReportType.ASYNC_PROCESSING, ReportFormat.MARKDOWN),
                (ReportType.STREAMING_GENERATION, ReportFormat.CSV)
            ]
            
            generated_reports = []
            
            for report_type, format in test_configs:
                config = ReportConfig(
                    report_type=report_type,
                    format=format,
                    time_range_hours=1,
                    include_charts=True,
                    include_trends=True
                )
                
                # Add custom test data
                custom_data = {
                    "test_component": {
                        "operations_count": 100,
                        "success_rate": 95.5,
                        "avg_response_time": 150.0
                    }
                }
                
                report_id = generator.generate_report(config, custom_data)
                generated_reports.append(report_id)
                
                print(f"   [+] Generated {report_type.value} report ({format.value}): {report_id}")
                
                # Verify report file exists
                report_info = generator.get_report_info(report_id)
                if report_info and os.path.exists(report_info.file_path):
                    file_size = os.path.getsize(report_info.file_path)
                    print(f"   [i] Report file: {report_info.file_path} ({file_size} bytes)")
                else:
                    print(f"   [!] Report file not found for {report_id}")
            
            # Test report listing
            recent_reports = generator.get_recent_reports(limit=5)
            print(f"   [i] Recent reports: {len(recent_reports)}")
            
            # Check statistics
            stats = generator.get_generator_statistics()
            print(f"   [i] Generator stats: {stats['reports_generated']} generated, {stats['avg_generation_time_ms']:.2f}ms avg")
            
            self.test_results['report_generator'] = len(generated_reports) > 0
            
        except Exception as e:
            print(f"   [!] ReportGenerator test failed: {e}")
            self.test_results['report_generator'] = False
    
    async def test_data_collector(self):
        """Test DataCollector functionality."""
        print("\\n[*] Testing DataCollector...")
        
        try:
            collector = get_data_collector()
            collector.start_collection()
            
            # Test data collection from different sources
            test_sources = [
                DataSource.ASYNC_PROCESSING,
                DataSource.STREAMING_GENERATION,
                DataSource.TELEMETRY,
                DataSource.SYSTEM_METRICS
            ]
            
            collected_data = {}
            
            for source in test_sources:
                data = collector.collect_data([source])
                collected_data.update(data)
                print(f"   [+] Collected data from {source.value}")
            
            # Test full system collection
            all_data = collector.collect_data()
            print(f"   [i] All data sources: {len(all_data)} sources")
            
            # Test convenience function
            system_metrics = collect_system_metrics()
            print(f"   [i] System metrics: {len(system_metrics)} categories")
            
            collector.stop_collection()
            print(f"   [+] Collection stopped")
            
            self.test_results['data_collector'] = len(collected_data) > 0
            
        except Exception as e:
            print(f"   [!] DataCollector test failed: {e}")
            self.test_results['data_collector'] = False
    
    async def test_dashboard_builder(self):
        """Test DashboardBuilder functionality."""
        print("\\n[*] Testing DashboardBuilder...")
        
        try:
            builder = get_dashboard_builder()
            
            # Test dashboard creation
            from testmaster.reporting.dashboard_builder import DashboardConfig, DashboardSection
            
            config = DashboardConfig(
                title="Test Performance Dashboard",
                sections=[
                    DashboardSection(
                        section_id="overview",
                        title="System Overview", 
                        chart_type=ChartType.LINE,
                        data_source="system",
                        metrics=["cpu", "memory"]
                    ),
                    DashboardSection(
                        section_id="performance",
                        title="Performance Metrics",
                        chart_type=ChartType.BAR,
                        data_source="performance", 
                        metrics=["throughput", "latency"]
                    )
                ],
                refresh_interval=30,
                theme="light"
            )
            
            dashboard_html = builder.build_dashboard(config)
            
            print(f"   [+] Built dashboard ({len(dashboard_html)} chars)")
            
            # Test convenience function
            perf_dashboard = build_performance_dashboard("Test Dashboard")
            print(f"   [+] Built performance dashboard ({len(perf_dashboard)} chars)")
            
            # Verify dashboard contains expected elements
            has_title = "Test Performance Dashboard" in dashboard_html or "Test Dashboard" in perf_dashboard
            has_sections = "section" in dashboard_html and "section" in perf_dashboard
            
            print(f"   [i] Dashboard contains title: {has_title}")
            print(f"   [i] Dashboard contains sections: {has_sections}")
            
            self.test_results['dashboard_builder'] = has_title and has_sections
            
        except Exception as e:
            print(f"   [!] DashboardBuilder test failed: {e}")
            self.test_results['dashboard_builder'] = False
    
    async def test_metrics_analyzer(self):
        """Test MetricsAnalyzer functionality."""
        print("\\n[*] Testing MetricsAnalyzer...")
        
        try:
            analyzer = get_metrics_analyzer()
            analyzer.start_analysis()
            
            # Test trend analysis with sample data
            sample_metrics = {
                "cpu_usage": [20.0, 25.0, 30.0, 35.0, 40.0],
                "memory_usage": [60.0, 58.0, 55.0, 52.0, 50.0],
                "response_time": [100, 105, 95, 110, 98],
                "throughput": [1000, 1050, 1100, 1150, 1200]
            }
            
            # Test trend analysis
            trends = analyzer.analyze_trends(sample_metrics)
            print(f"   [+] Analyzed trends: {len(trends)} metrics")
            
            for trend in trends:
                print(f"   [i] {trend.metric_name}: {trend.trend_direction} (confidence: {trend.confidence:.2f})")
            
            # Test anomaly detection
            anomalies = analyzer.detect_anomalies(sample_metrics)
            print(f"   [+] Detected anomalies: {len(anomalies)} metrics")
            
            # Test convenience function
            analysis_result = analyze_performance_metrics(sample_metrics)
            print(f"   [i] Analysis result: {len(analysis_result['trends'])} trends, {len(analysis_result['anomalies'])} anomalies")
            
            analyzer.shutdown()
            print(f"   [+] Analysis stopped")
            
            self.test_results['metrics_analyzer'] = len(trends) > 0
            
        except Exception as e:
            print(f"   [!] MetricsAnalyzer test failed: {e}")
            self.test_results['metrics_analyzer'] = False
    
    async def test_report_scheduler(self):
        """Test ReportScheduler functionality."""
        print("\\n[*] Testing ReportScheduler...")
        
        try:
            scheduler = get_report_scheduler()
            scheduler.start_scheduler()
            
            # Test scheduling configuration
            from testmaster.reporting.report_scheduler import ScheduleConfig
            
            schedule_config = ScheduleConfig(
                name="Daily System Report",
                interval_hours=24,
                report_type="system_overview",
                delivery_method=DeliveryMethod.FILE_SYSTEM,
                recipients=["admin@testmaster.com"]
            )
            
            schedule_id = scheduler.schedule_report(schedule_config)
            print(f"   [+] Scheduled report: {schedule_id}")
            
            # Test convenience function
            auto_schedule_id = schedule_automated_reports("Automated Reports", interval_hours=12)
            print(f"   [+] Automated schedule: {auto_schedule_id}")
            
            # Check scheduler status
            print(f"   [i] Scheduler running: {scheduler.is_running}")
            print(f"   [i] Scheduled reports: {len(scheduler.scheduled_reports)}")
            
            scheduler.shutdown()
            print(f"   [+] Scheduler stopped")
            
            self.test_results['report_scheduler'] = len(schedule_id) > 0
            
        except Exception as e:
            print(f"   [!] ReportScheduler test failed: {e}")
            self.test_results['report_scheduler'] = False
    
    async def test_integration(self):
        """Test integrated reporting functionality."""
        print("\\n[*] Testing Integration...")
        
        try:
            # Test end-to-end reporting workflow
            print("   [>] Starting integrated reporting workflow...")
            
            # 1. Collect system data
            data = collect_system_metrics()
            print(f"   [+] Collected system data: {len(data)} sources")
            
            # 2. Analyze metrics
            analysis = analyze_performance_metrics({"test_metric": [1, 2, 3, 4, 5]})
            print(f"   [+] Analyzed metrics: {len(analysis['trends'])} trends")
            
            # 3. Generate comprehensive report
            report_id = generate_performance_report(
                report_type=ReportType.SYSTEM_OVERVIEW,
                format=ReportFormat.HTML,
                time_range_hours=1,
                custom_data={
                    "integration_test": True,
                    "collected_data": data,
                    "analysis_results": analysis
                }
            )
            print(f"   [+] Generated integrated report: {report_id}")
            
            # 4. Build dashboard
            dashboard = build_performance_dashboard("Integration Test Dashboard")
            print(f"   [+] Built integration dashboard ({len(dashboard)} chars)")
            
            # 5. Schedule automated reporting
            schedule_id = schedule_automated_reports("Integration Test Reports")
            print(f"   [+] Scheduled automated reports: {schedule_id}")
            
            # Verify integration results
            generator = get_report_generator()
            report_info = generator.get_report_info(report_id)
            
            integration_success = (
                report_info is not None and
                os.path.exists(report_info.file_path) and
                len(dashboard) > 0 and
                len(schedule_id) > 0
            )
            
            # Check overall system statistics
            gen_stats = generator.get_generator_statistics()
            print(f"   [i] Integration stats:")
            print(f"      - Reports generated: {gen_stats['reports_generated']}")
            print(f"      - Average generation time: {gen_stats['avg_generation_time_ms']:.2f}ms")
            print(f"      - Output directory: {gen_stats['output_directory']}")
            
            self.test_results['integration'] = integration_success
            
        except Exception as e:
            print(f"   [!] Integration test failed: {e}")
            self.test_results['integration'] = False
    
    def display_results(self):
        """Display test results summary."""
        print("\\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for component, result in self.test_results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{component.replace('_', ' ').title()}: {status}")
        
        print(f"\\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All reporting system tests PASSED!")
        else:
            print("Some tests failed - check implementation")
        
        execution_time = time.time() - self.start_time
        print(f"Total execution time: {execution_time:.2f} seconds")
        
        # Show generated reports
        if self.test_output_dir.exists():
            report_files = list(self.test_output_dir.glob("*"))
            print(f"\\nGenerated {len(report_files)} report files:")
            for file_path in report_files[:5]:  # Show first 5
                print(f"  - {file_path.name}")
            if len(report_files) > 5:
                print(f"  ... and {len(report_files) - 5} more")

async def main():
    """Main test execution."""
    try:
        # Run tests
        test_suite = ReportingSystemTest()
        await test_suite.run_all_tests()
        
    finally:
        # Cleanup
        print("\\nCleaning up reporting system...")
        shutdown_reporting()
        print("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())