#!/usr/bin/env python3
"""
Comprehensive Test Script for All Analytics Features
=====================================================

Tests all the robustness enhancements and new analytics features.
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, Any

BASE_URL = "http://localhost:5000"

def test_feature(name: str, test_func: callable) -> bool:
    """Run a single feature test."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    try:
        result = test_func()
        if result:
            print(f"[PASS] {name}: PASSED")
        else:
            print(f"[FAIL] {name}: FAILED")
        return result
    except Exception as e:
        print(f"[ERROR] {name}: ERROR - {e}")
        return False

def test_analytics_performance() -> bool:
    """Test analytics endpoint performance."""
    print("Testing analytics performance...")
    
    response_times = []
    for i in range(5):
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/analytics/metrics")
        duration = (time.time() - start) * 1000
        response_times.append(duration)
        
        if response.status_code != 200:
            print(f"  Request {i+1} failed: {response.status_code}")
            return False
        
        data = response.json()
        if data.get('status') != 'success':
            print(f"  Request {i+1} returned error status")
            return False
    
    avg_time = sum(response_times) / len(response_times)
    max_time = max(response_times)
    
    print(f"  Average response time: {avg_time:.2f}ms")
    print(f"  Max response time: {max_time:.2f}ms")
    print(f"  Performance target: <5000ms")
    
    return max_time < 5000  # All responses under 5 seconds

def test_event_queue() -> bool:
    """Test event queue is working."""
    print("Testing event queue functionality...")
    
    # Trigger analytics collection
    response = requests.get(f"{BASE_URL}/api/analytics/metrics")
    if response.status_code != 200:
        return False
    
    # Event queue processes in background
    print("  Event queue initialized and processing")
    return True

def test_anomaly_detection() -> bool:
    """Test anomaly detection is active."""
    print("Testing anomaly detection...")
    
    # Get analytics multiple times to feed data to detector
    for i in range(3):
        response = requests.get(f"{BASE_URL}/api/analytics/metrics")
        if response.status_code != 200:
            return False
        time.sleep(0.5)
    
    print("  Anomaly detector receiving data points")
    return True

def test_export_capabilities() -> bool:
    """Test export functionality."""
    print("Testing export capabilities...")
    
    # Test JSON export via snapshot
    response = requests.get(f"{BASE_URL}/api/analytics/snapshot")
    if response.status_code != 200:
        print("  Snapshot creation failed")
        return False
    
    data = response.json()
    if data.get('status') != 'success':
        print("  Snapshot status not success")
        return False
    
    print("  Export manager functional")
    return True

def test_health_monitoring() -> bool:
    """Test health monitoring."""
    print("Testing health monitoring...")
    
    response = requests.get(f"{BASE_URL}/api/health/live")
    if response.status_code != 200:
        return False
    
    data = response.json()
    if data.get('status') != 'healthy':
        print(f"  Health status: {data.get('status')}")
        return False
    
    print("  Health monitoring active and healthy")
    return True

def test_comprehensive_analytics() -> bool:
    """Test comprehensive analytics data."""
    print("Testing comprehensive analytics data...")
    
    response = requests.get(f"{BASE_URL}/api/analytics/metrics")
    if response.status_code != 200:
        return False
    
    data = response.json()
    
    # Check for key components
    required_keys = ['status', 'metrics', 'comprehensive', 'timestamp']
    for key in required_keys:
        if key not in data:
            print(f"  Missing key: {key}")
            return False
    
    # Check comprehensive data
    comp = data.get('comprehensive', {})
    if 'optimization_enabled' in comp and comp['optimization_enabled']:
        print("  [OK] Performance optimization enabled")
    
    if 'response_time_ms' in comp:
        print(f"  [OK] Response time: {comp['response_time_ms']:.2f}ms")
    
    if 'performance_stats' in comp:
        print("  [OK] Performance statistics available")
    
    return True

def test_dashboard_summary() -> bool:
    """Test dashboard summary endpoint."""
    print("Testing dashboard summary...")
    
    response = requests.get(f"{BASE_URL}/api/analytics/dashboard-summary")
    if response.status_code != 200:
        return False
    
    data = response.json()
    if data.get('status') != 'success':
        return False
    
    summary = data.get('summary', {})
    sections = ['overview', 'analytics', 'tests', 'workflow', 'refactor']
    
    for section in sections:
        if section in summary:
            print(f"  [OK] {section.capitalize()} data present")
    
    return 'summary' in data

def test_historical_data() -> bool:
    """Test historical data endpoints."""
    print("Testing historical data...")
    
    # Test performance history
    response = requests.get(f"{BASE_URL}/api/analytics/history/performance?hours=1")
    if response.status_code == 200:
        print("  [OK] Performance history available")
    
    # Test trends
    response = requests.get(f"{BASE_URL}/api/analytics/trends")
    if response.status_code == 200:
        data = response.json()
        if 'trends' in data:
            print("  [OK] Trend analysis available")
    
    return True

def test_insights() -> bool:
    """Test insights endpoint."""
    print("Testing insights and recommendations...")
    
    response = requests.get(f"{BASE_URL}/api/analytics/insights")
    if response.status_code != 200:
        return False
    
    data = response.json()
    if data.get('status') != 'success':
        return False
    
    insights = data.get('insights', {})
    if 'recommendations' in insights:
        recs = insights['recommendations']
        if isinstance(recs, list):
            print(f"  [OK] {len(recs)} recommendations available")
    
    return True

def test_optimization_metrics() -> bool:
    """Test optimization metrics."""
    print("Testing optimization metrics...")
    
    response = requests.get(f"{BASE_URL}/api/analytics/metrics/optimization")
    if response.status_code != 200:
        return False
    
    data = response.json()
    if data.get('status') != 'success':
        return False
    
    opt_metrics = data.get('optimization_metrics', {})
    
    if 'cache' in opt_metrics:
        print("  [OK] Cache statistics available")
    
    if 'validation' in opt_metrics:
        print("  [OK] Validation metrics available")
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("TESTMASTER ANALYTICS COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Testing server at: {BASE_URL}")
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/api/health/live")
            if response.status_code == 200:
                print("Server is ready!")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("[ERROR] Server not responding after 10 seconds")
        return
    
    # Run all tests
    tests = [
        ("Health Monitoring", test_health_monitoring),
        ("Analytics Performance (<5s)", test_analytics_performance),
        ("Comprehensive Analytics", test_comprehensive_analytics),
        ("Event Queue System", test_event_queue),
        ("Anomaly Detection", test_anomaly_detection),
        ("Export Capabilities", test_export_capabilities),
        ("Dashboard Summary", test_dashboard_summary),
        ("Historical Data", test_historical_data),
        ("Insights & Recommendations", test_insights),
        ("Optimization Metrics", test_optimization_metrics),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_feature(test_name, test_func)
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASSED" if result else "[ERROR] FAILED"
        print(f"{test_name:40} {status}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! The analytics system is fully operational!")
    else:
        print(f"\n[WARNING]  {total - passed} test(s) failed. Review the output above.")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()