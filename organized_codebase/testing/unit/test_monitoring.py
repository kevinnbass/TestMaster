#!/usr/bin/env python3
"""
Quick test for performance monitoring infrastructure
"""

import time
import requests
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.monitoring.cc_1.performance_monitoring_infrastructure import PerformanceMonitoringSystem, MonitoringConfig

def main():
    print("Testing Performance Monitoring Infrastructure")
    print("=" * 50)
    
    # Simple config for testing
    config = MonitoringConfig(
        metrics_port=9091,  # Different port to avoid conflicts
        collection_interval=1.0,
        alert_thresholds={'cpu_usage_percent': 90.0},  # High threshold
        alert_channels=['console'],
        enable_prometheus=True,
        enable_alerting=True
    )
    
    # Initialize monitoring
    monitoring = PerformanceMonitoringSystem(config)
    
    try:
        # Start monitoring
        monitoring.start()
        print("Monitoring started successfully!")
        
        # Wait for some metrics collection
        time.sleep(5)
        
        # Check status
        status = monitoring.get_system_status()
        print(f"\nSystem Status: {status['system_health']}")
        print(f"Active Metrics: {status['metrics_count']}")
        print(f"Active Alerts: {status['active_alerts']}")
        
        # Test Prometheus endpoint
        try:
            response = requests.get(f"http://localhost:{config.metrics_port}/metrics", timeout=2)
            if response.status_code == 200:
                print("Prometheus endpoint working!")
                print(f"Metrics data: {len(response.text)} characters")
            else:
                print(f"Prometheus endpoint error: {response.status_code}")
        except Exception as e:
            print(f"Could not test Prometheus endpoint: {e}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    finally:
        monitoring.stop()

if __name__ == "__main__":
    main()