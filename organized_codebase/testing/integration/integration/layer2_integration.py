"""
TestMaster Layer 2 Integration Example

Complete integration example showing how all Layer 2 components
work together for active monitoring and communication.

This demonstrates:
- File monitoring with idle detection
- Test scheduling with queue management
- Claude Code communication
- Real-time dashboard with alerts
- Metrics display and trend analysis
"""

import time
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Layer 2 imports
from .monitoring import FileWatcher, IdleDetector, TestScheduler
from .communication import ClaudeMessenger, TagReader, MessageQueue
from .ui import TestMasterDashboard, MetricsDisplay, AlertSystem
from .core.layer_manager import LayerManager

# Helper functions for the integration
from .communication.claude_messenger import (
    create_test_failure_info, 
    create_module_attention_info,
    create_coverage_gap_info
)
from .ui.alert_system import (
    create_test_failure_alert,
    create_idle_module_alert, 
    create_coverage_alert
)


class TestMasterLayer2:
    """
    Complete Layer 2 integration - Active Monitoring & Communication.
    
    Orchestrates all Layer 2 components:
    - Real-time file monitoring
    - Idle detection with 2-hour threshold
    - Test scheduling and execution
    - Claude Code communication
    - Live dashboard with metrics
    - Alert and notification system
    """
    
    def __init__(self, source_dir: str, test_dir: str,
                 dashboard_port: int = 8080,
                 enable_dashboard: bool = True):
        """
        Initialize Layer 2 system.
        
        Args:
            source_dir: Source code directory to monitor
            test_dir: Test directory
            dashboard_port: Port for web dashboard
            enable_dashboard: Whether to start dashboard
        """
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.enable_dashboard = enable_dashboard
        
        # Initialize layer manager and validate Layer 2 is enabled
        self.layer_manager = LayerManager()
        if not self.layer_manager.is_enabled("layer2_monitoring"):
            raise RuntimeError("Layer 2 monitoring must be enabled in testmaster_config.yaml")
        
        print("🚀 Initializing TestMaster Layer 2...")
        
        # Initialize monitoring components
        self.file_watcher = FileWatcher(
            watch_paths=[str(self.source_dir), str(self.test_dir)]
        )
        
        self.idle_detector = IdleDetector(
            watch_paths=[str(self.source_dir)],
            idle_threshold_hours=2.0,
            scan_interval_minutes=30.0
        )
        
        self.test_scheduler = TestScheduler(
            max_workers=2,
            max_queue_size=100
        )
        
        # Initialize communication components
        self.claude_messenger = ClaudeMessenger()
        self.tag_reader = TagReader([str(self.source_dir), str(self.test_dir)])
        self.message_queue = MessageQueue()
        
        # Initialize UI components
        self.alert_system = AlertSystem()
        self.metrics_display = MetricsDisplay()
        
        if self.enable_dashboard:
            from .ui.dashboard import DashboardConfig
            dashboard_config = DashboardConfig(port=dashboard_port)
            self.dashboard = TestMasterDashboard(dashboard_config)
        else:
            self.dashboard = None
        
        # Setup callbacks and integration
        self._setup_integration_callbacks()
        
        # System state
        self._is_running = False
        self._stats = {
            'start_time': None,
            'files_monitored': 0,
            'tests_executed': 0,
            'alerts_created': 0,
            'messages_sent': 0
        }
        
        print("✅ Layer 2 system initialized")
    
    def start(self, open_dashboard: bool = True):
        """
        Start all Layer 2 components.
        
        Args:
            open_dashboard: Whether to open browser for dashboard
        """
        if self._is_running:
            print("⚠️ Layer 2 system is already running")
            return
        
        print("🚀 Starting TestMaster Layer 2 system...")
        
        # Start core components
        self.alert_system.start()
        self.message_queue.start()
        self.claude_messenger.start_monitoring()
        
        # Start monitoring components
        self.file_watcher.start()
        self.idle_detector.start_monitoring()
        self.test_scheduler.start()
        
        # Start dashboard
        if self.dashboard:
            self.dashboard.start(open_browser=open_dashboard)
        
        # Initial system scan
        self._perform_initial_scan()
        
        self._is_running = True
        self._stats['start_time'] = datetime.now()
        
        print("✅ Layer 2 system started successfully")
        
        if self.dashboard:
            print(f"🌐 Dashboard available at: {self.dashboard.get_dashboard_url()}")
        
        # Send startup notification
        self._send_startup_notification()
    
    def stop(self):
        """Stop all Layer 2 components."""
        if not self._is_running:
            return
        
        print("🛑 Stopping TestMaster Layer 2 system...")
        
        # Stop monitoring components
        self.file_watcher.stop()
        self.idle_detector.stop_monitoring()
        self.test_scheduler.stop()
        
        # Stop communication components
        self.claude_messenger.stop_monitoring()
        self.message_queue.stop()
        
        # Stop UI components
        self.alert_system.stop()
        if self.dashboard:
            self.dashboard.stop()
        
        self._is_running = False
        
        # Send shutdown notification
        self._send_shutdown_notification()
        
        print("✅ Layer 2 system stopped")
    
    def _setup_integration_callbacks(self):
        """Setup callbacks for component integration."""
        
        # File watcher callbacks
        def on_file_changed(event):
            self._stats['files_monitored'] += 1
            
            # Update dashboard
            if self.dashboard:
                self.dashboard.add_alert(
                    f"File changed: {Path(event.file_path).name}",
                    level="info"
                )
            
            # Schedule related tests if source file changed
            if self._is_source_file(event.file_path):
                self._schedule_related_tests(event.file_path)
        
        self.file_watcher.callbacks.on_file_modified = on_file_changed
        
        # Idle detector callbacks
        def on_module_idle(idle_module):
            # Create alert
            alert_id = create_idle_module_alert(
                idle_module.module_path,
                idle_module.idle_duration.total_seconds() / 3600,
                self.alert_system
            )
            
            # Send to Claude Code
            idle_info = create_module_attention_info(
                path=idle_module.module_path,
                status="idle_2_hours",
                coverage=getattr(idle_module, 'test_coverage', None),
                risks=["Stale code", "No recent changes"],
                recommendation="Review for updates or testing needs"
            )
            
            self.claude_messenger.send_idle_module_alert([idle_info])
            
            # Update dashboard
            if self.dashboard:
                self.dashboard.update_idle_modules([{
                    'module_path': idle_module.module_path,
                    'idle_duration_hours': idle_module.idle_duration.total_seconds() / 3600
                }])
        
        self.idle_detector.on_module_idle = on_module_idle
        
        # Test scheduler callbacks
        def on_test_completed(test):
            self._stats['tests_executed'] += 1
            
            # Update dashboard
            if self.dashboard:
                self.dashboard.add_test_result({
                    'test_name': Path(test.test_path).name,
                    'status': 'passed' if test.result_code == 0 else 'failed',
                    'duration': test.actual_duration
                })
            
            # Create alert if test failed
            if test.result_code != 0:
                alert_id = create_test_failure_alert(
                    test.test_path,
                    test.error_output or "Test failed",
                    self.alert_system
                )
                
                # Send to Claude Code
                failure_info = create_test_failure_info(
                    module=test.test_path,
                    test=Path(test.test_path).name,
                    failure=test.error_output or "Unknown failure",
                    suggested_action="Review test failure and fix issues"
                )
                
                self.claude_messenger.send_breaking_test_alert([failure_info])
        
        self.test_scheduler.on_test_completed = on_test_completed
        self.test_scheduler.on_test_failed = on_test_completed  # Same handler for both
        
        # Claude messenger callbacks
        def on_directive_received(directive):
            # Process Claude Code directives
            self._process_claude_directive(directive)
        
        self.claude_messenger.on_directive_received = on_directive_received
        
        # Alert system callbacks
        def on_alert_created(alert):
            self._stats['alerts_created'] += 1
            
            # Send to dashboard
            if self.dashboard:
                self.dashboard.add_alert(
                    alert.message,
                    level=alert.level.name.lower(),
                    metadata=alert.metadata
                )
        
        self.alert_system.on_alert_created = on_alert_created
    
    def _perform_initial_scan(self):
        """Perform initial system scan."""
        print("🔍 Performing initial system scan...")
        
        # Scan for tags
        tag_results = self.tag_reader.scan_all_files()
        print(f"📝 Found tags in {len(tag_results)} files")
        
        # Scan for idle modules
        idle_stats = self.idle_detector.scan_for_idle_modules()
        print(f"😴 Found {idle_stats.idle_modules} idle modules")
        
        # Update dashboard with initial data
        if self.dashboard:
            self.dashboard.update_system_status({
                'layer2_monitoring': 'active',
                'files_with_tags': len(tag_results),
                'idle_modules': idle_stats.idle_modules,
                'monitoring_start': datetime.now().isoformat()
            })
    
    def _schedule_related_tests(self, source_file: str):
        """Schedule tests related to a changed source file."""
        # Find tests that cover this module
        covering_tests = self.tag_reader.get_test_modules_covering(source_file)
        
        for test_module in covering_tests:
            if Path(test_module.file_path).exists():
                # Schedule with high priority since source changed
                from .monitoring.test_scheduler import TestPriority
                self.test_scheduler.schedule_test(
                    test_module.file_path,
                    priority=TestPriority.HIGH,
                    metadata={'triggered_by': source_file}
                )
    
    def _process_claude_directive(self, directive):
        """Process a directive from Claude Code."""
        print(f"📥 Processing Claude directive: {directive.directive_id}")
        
        # Process monitor priority changes
        for priority_item in directive.monitor_priority:
            path = priority_item.get('path')
            level = priority_item.get('level', 'NORMAL')
            
            if path:
                print(f"📌 Setting monitor priority for {path}: {level}")
                # Could update file watcher priorities here
        
        # Process temporary ignore directives
        for ignore_item in directive.temporary_ignore:
            path = ignore_item.get('path')
            until = ignore_item.get('until')
            
            if path and until:
                print(f"🔇 Temporarily ignoring {path} until {until}")
                # Could add to ignore list
        
        # Process test preferences
        for test_pref in directive.test_preferences:
            pattern = test_pref.get('module_pattern')
            test_style = test_pref.get('test_style')
            
            if pattern and test_style:
                print(f"⚙️ Test preference: {pattern} -> {test_style}")
                # Could update test scheduling preferences
    
    def _is_source_file(self, file_path: str) -> bool:
        """Check if file is a source file (not test file)."""
        path = Path(file_path)
        name = path.name.lower()
        
        # Check if it's in source directory and not a test file
        return (str(self.source_dir) in str(path) and 
                path.suffix == '.py' and
                not name.startswith('test_') and
                not name.endswith('_test.py'))
    
    def _send_startup_notification(self):
        """Send startup notification to Claude Code."""
        startup_status = {
            'system': 'TestMaster Layer 2',
            'status': 'started',
            'components': [
                'file_monitoring',
                'idle_detection', 
                'test_scheduling',
                'claude_communication',
                'alert_system'
            ],
            'monitoring_paths': [str(self.source_dir), str(self.test_dir)],
            'dashboard_url': self.dashboard.get_dashboard_url() if self.dashboard else None
        }
        
        self.claude_messenger.send_system_alert(
            "TestMaster Layer 2 started successfully",
            metadata=startup_status
        )
        
        self._stats['messages_sent'] += 1
    
    def _send_shutdown_notification(self):
        """Send shutdown notification to Claude Code."""
        uptime = datetime.now() - self._stats['start_time'] if self._stats['start_time'] else timedelta(0)
        
        shutdown_status = {
            'system': 'TestMaster Layer 2',
            'status': 'stopped',
            'uptime_minutes': uptime.total_seconds() / 60,
            'stats': dict(self._stats)
        }
        
        self.claude_messenger.send_system_alert(
            "TestMaster Layer 2 stopped",
            metadata=shutdown_status
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = datetime.now() - self._stats['start_time'] if self._stats['start_time'] else timedelta(0)
        
        return {
            'layer2_status': 'running' if self._is_running else 'stopped',
            'uptime_minutes': uptime.total_seconds() / 60,
            'components': {
                'file_watcher': self.file_watcher.is_running,
                'idle_detector': self.idle_detector._is_scanning,
                'test_scheduler': self.test_scheduler._is_running,
                'claude_messenger': self.claude_messenger._monitoring,
                'message_queue': self.message_queue._is_running,
                'alert_system': self.alert_system._is_running,
                'dashboard': self.dashboard._is_running if self.dashboard else False
            },
            'statistics': dict(self._stats),
            'queue_status': self.test_scheduler.get_queue_status(),
            'alert_stats': self.alert_system.get_alert_statistics().__dict__,
            'communication_stats': self.claude_messenger.get_communication_statistics()
        }
    
    def send_test_coverage_update(self, coverage_data: Dict[str, Any]):
        """Send test coverage update to dashboard and Claude Code."""
        # Update metrics display
        self.metrics_display.update_coverage_metrics(coverage_data)
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.update_coverage_data(coverage_data)
        
        # Check for coverage gaps and alert if needed
        overall_coverage = coverage_data.get('overall_coverage', 0)
        if overall_coverage < 70:  # Threshold
            create_coverage_alert(overall_coverage, 70, self.alert_system)
            
            # Send coverage gaps to Claude Code
            gaps = []
            for file_path, file_coverage in coverage_data.get('file_coverage', {}).items():
                if file_coverage < 70:
                    gap_info = create_coverage_gap_info(
                        module=file_path,
                        critical_paths=file_coverage < 50,
                        suggested_tests=[f"Add tests for {Path(file_path).name}"]
                    )
                    gaps.append(gap_info)
            
            if gaps:
                self.claude_messenger.send_coverage_gap_report(gaps)
    
    def force_idle_scan(self):
        """Force an immediate idle module scan."""
        return self.idle_detector.scan_for_idle_modules()
    
    def schedule_test_batch(self, test_paths: List[str]):
        """Schedule a batch of tests for execution."""
        return self.test_scheduler.schedule_multiple_tests(test_paths)
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL if available."""
        if self.dashboard:
            return self.dashboard.get_dashboard_url()
        return "Dashboard not enabled"


# Example usage function
def run_layer2_example():
    """
    Example of running Layer 2 system.
    
    This would typically be called from a main script or CLI command.
    """
    # Initialize Layer 2 system
    layer2 = TestMasterLayer2(
        source_dir="./src",
        test_dir="./tests",
        dashboard_port=8080,
        enable_dashboard=True
    )
    
    try:
        # Start the system
        layer2.start(open_dashboard=True)
        
        print("🎯 TestMaster Layer 2 is now running!")
        print("📊 Real-time monitoring active")
        print("🌐 Dashboard available")
        print("📡 Claude Code communication enabled")
        print()
        print("Press Ctrl+C to stop...")
        
        # Keep running until interrupted
        while True:
            time.sleep(10)
            
            # Periodic status update
            status = layer2.get_system_status()
            print(f"⚡ Status: {status['components']} | Stats: {status['statistics']}")
            
    except KeyboardInterrupt:
        print("\\n🛑 Stopping Layer 2 system...")
        layer2.stop()
        print("✅ Layer 2 stopped successfully")


if __name__ == "__main__":
    run_layer2_example()