#!/usr/bin/env python3
"""
TestMaster Production Monitoring Deployment Script

Easy deployment script for starting TestMaster real-time monitoring in production.
Supports both console and web-based monitoring with automatic service management.
"""

import sys
import os
import argparse
import subprocess
import time
import signal
import atexit
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = ['flask', 'flask_cors']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("Missing required dependencies for web monitoring:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nInstall with: pip install flask flask-cors")
        return False
    
    return True

def start_console_monitoring(mode='full', interval=1.0):
    """Start console-based monitoring."""
    print("=" * 80)
    print("Starting TestMaster Console Monitoring")
    print("=" * 80)
    
    script_path = Path(__file__).parent / "real_time_monitor.py"
    cmd = [sys.executable, str(script_path), "--mode", mode, "--interval", str(interval)]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nConsole monitoring stopped")

def start_web_monitoring(port=5000, host='0.0.0.0', debug=False):
    """Start web-based monitoring."""
    if not check_dependencies():
        print("Cannot start web monitoring without required dependencies")
        return False
    
    print("=" * 80)
    print("Starting TestMaster Web Monitoring Dashboard")
    print("=" * 80)
    print(f"Dashboard will be available at: http://{host}:{port}")
    print("API endpoints:")
    print(f"  - Metrics: http://{host}:{port}/api/metrics")
    print(f"  - Alerts: http://{host}:{port}/api/alerts")
    print(f"  - Components: http://{host}:{port}/api/components")
    print(f"  - Health: http://{host}:{port}/api/health")
    print("=" * 80)
    
    script_path = Path(__file__).parent / "web_monitor.py"
    cmd = [sys.executable, str(script_path), "--port", str(port), "--host", host]
    
    if debug:
        cmd.append("--debug")
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\nWeb monitoring stopped")
        return False

def export_metrics(output_file):
    """Export current metrics to file."""
    print(f"Exporting metrics to {output_file}...")
    
    script_path = Path(__file__).parent / "real_time_monitor.py"
    cmd = [sys.executable, str(script_path), "--export", output_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[SUCCESS] Metrics successfully exported to {output_file}")
            return True
        else:
            print(f"[FAIL] Export failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[FAIL] Export error: {e}")
        return False

def run_integration_check():
    """Run integration check on monitoring components."""
    print("=" * 80)
    print("TestMaster Monitoring System - Integration Check")
    print("=" * 80)
    
    # Test basic imports
    try:
        from real_time_monitor import RealTimeMonitor, MonitoringMode
        print("[PASS] Real-time monitor import successful")
    except Exception as e:
        print(f"[FAIL] Real-time monitor import failed: {e}")
        return False
    
    # Test web monitor imports (if dependencies available)
    if check_dependencies():
        try:
            from web_monitor import WebMonitoringServer
            print("[PASS] Web monitor import successful")
        except Exception as e:
            print(f"[FAIL] Web monitor import failed: {e}")
    else:
        print("[WARN] Web monitor dependencies not available")
    
    # Test TestMaster components
    components_tested = 0
    components_working = 0
    
    test_components = [
        ("Core Orchestrator", "testmaster.core.orchestrator", "WorkflowDAG"),
        ("Shared State", "testmaster.core.shared_state", "SharedState"),
        ("Config Intelligence", "testmaster.core.config", "ConfigurationIntelligenceAgent"),
        ("Hierarchical Planning", "testmaster.intelligence.hierarchical_planning.htp_reasoning", "PlanGenerator"),
        ("Consensus Engine", "testmaster.intelligence.consensus.consensus_engine", "ConsensusEngine"),
        ("Security Intelligence", "testmaster.intelligence.security.security_intelligence_agent", "SecurityIntelligenceAgent"),
        ("Protocol Bridge", "testmaster.intelligence.bridges.protocol_communication_bridge", "ProtocolCommunicationBridge"),
        ("Event Bridge", "testmaster.intelligence.bridges.event_monitoring_bridge", "EventMonitoringBridge"),
    ]
    
    for name, module_path, class_name in test_components:
        components_tested += 1
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"[PASS] {name} component available")
            components_working += 1
        except Exception as e:
            print(f"[FAIL] {name} component failed: {e}")
    
    print("=" * 80)
    print(f"Integration Check Complete: {components_working}/{components_tested} components working")
    
    if components_working >= components_tested * 0.8:  # 80% success rate
        print("[SUCCESS] Monitoring system is ready for production use!")
        return True
    else:
        print("[WARNING] Some components have issues - monitoring may have limited functionality")
        return False

def show_monitoring_status():
    """Show current monitoring status and available options."""
    print("=" * 80)
    print("TestMaster Hybrid Intelligence Platform - Monitoring Options")
    print("=" * 80)
    print()
    
    print("AVAILABLE MONITORING MODES:")
    print("-" * 40)
    print("1. Console Dashboard - Full real-time display")
    print("2. Web Dashboard - Professional web interface")
    print("3. Alerts Only - Monitor alerts in console")
    print("4. Metrics Export - Export current metrics to file")
    print("5. Integration Check - Verify all components")
    print()
    
    print("PRODUCTION DEPLOYMENT EXAMPLES:")
    print("-" * 40)
    print("# Start web dashboard (recommended for production)")
    print("python start_monitoring.py --web --port 8080")
    print()
    print("# Start console monitoring")
    print("python start_monitoring.py --console --mode full")
    print()
    print("# Export metrics for external monitoring")
    print("python start_monitoring.py --export system_metrics.json")
    print()
    print("# Run integration check")
    print("python start_monitoring.py --check")
    print()
    
    print("CONFIGURATION PROFILES:")
    print("-" * 40)
    print("Set TESTMASTER_PROFILE environment variable:")
    print("  - development (default)")
    print("  - production (optimized for performance)")  
    print("  - security_focused (enhanced security)")
    print("  - high_performance (maximum throughput)")
    print()
    
    print("WEB API ENDPOINTS:")
    print("-" * 40)
    print("When web monitoring is running:")
    print("  - GET /api/metrics - Current system metrics")
    print("  - GET /api/metrics/history - Historical data")
    print("  - GET /api/components - Component status")
    print("  - GET /api/alerts - Active alerts")
    print("  - GET /api/health - Health check")
    print("  - GET /api/config - Configuration info")
    print("=" * 80)

def main():
    """Main entry point for monitoring deployment."""
    parser = argparse.ArgumentParser(
        description="TestMaster Production Monitoring Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_monitoring.py --web --port 8080          # Web dashboard on port 8080
  python start_monitoring.py --console --mode full      # Full console monitoring
  python start_monitoring.py --export metrics.json      # Export metrics
  python start_monitoring.py --check                    # Integration check
  python start_monitoring.py --status                   # Show options
        """
    )
    
    # Monitoring mode selection
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--web', action='store_true', help='Start web-based monitoring dashboard')
    mode_group.add_argument('--console', action='store_true', help='Start console-based monitoring')
    mode_group.add_argument('--export', type=str, metavar='FILE', help='Export metrics to JSON file')
    mode_group.add_argument('--check', action='store_true', help='Run integration check')
    mode_group.add_argument('--status', action='store_true', help='Show monitoring status and options')
    
    # Console monitoring options
    parser.add_argument('--mode', choices=['dashboard', 'alerts_only', 'performance', 'security', 'full'], 
                       default='full', help='Console monitoring mode')
    parser.add_argument('--interval', type=float, default=1.0, help='Update interval in seconds')
    
    # Web monitoring options
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for web server')
    
    args = parser.parse_args()
    
    # If no specific mode selected, show status
    if not any([args.web, args.console, args.export, args.check, args.status]):
        show_monitoring_status()
        return
    
    # Handle specific modes
    if args.status:
        show_monitoring_status()
    elif args.check:
        run_integration_check()
    elif args.export:
        export_metrics(args.export)
    elif args.web:
        start_web_monitoring(args.port, args.host, args.debug)
    elif args.console:
        start_console_monitoring(args.mode, args.interval)

if __name__ == "__main__":
    main()