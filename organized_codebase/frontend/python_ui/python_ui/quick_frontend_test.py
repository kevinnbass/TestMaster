#!/usr/bin/env python3
"""
Quick Frontend Integration Test
================================

Demonstrates how we test frontend-backend integration without a browser.
This is a faster, streamlined version that shows the key testing concepts.

Author: TestMaster Team
"""

import requests
import json
import time
from datetime import datetime
import concurrent.futures

def test_frontend_integration():
    """Quick test of frontend-backend integration."""
    print("=" * 70)
    print("AUTOMATED FRONTEND TESTING (NO BROWSER REQUIRED)")
    print("=" * 70)
    print()
    
    base_url = "http://localhost:5000"
    
    # 1. TEST API ENDPOINTS
    print("1. TESTING API ENDPOINTS FOR FRONTEND")
    print("-" * 40)
    
    critical_endpoints = [
        '/api/health/live',
        '/api/analytics/summary',
        '/api/performance/metrics',
        '/api/monitoring/robustness'
    ]
    
    working_endpoints = 0
    chart_ready_endpoints = 0
    
    for endpoint in critical_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=3)
            if response.status_code == 200:
                data = response.json()
                working_endpoints += 1
                
                # Check if data is chart-ready for frontend
                has_charts = 'charts' in data
                has_timestamp = 'timestamp' in data
                has_status = 'status' in data
                
                is_chart_ready = has_charts and has_timestamp
                if is_chart_ready:
                    chart_ready_endpoints += 1
                
                status = "CHART-READY" if is_chart_ready else "BASIC"
                print(f"  {endpoint:35} [{status}] {len(response.text)} bytes")
                
                # Show available chart types
                if has_charts:
                    chart_types = list(data['charts'].keys())[:3]
                    print(f"    Charts: {', '.join(chart_types)}")
            else:
                print(f"  {endpoint:35} [ERROR] Status {response.status_code}")
        except Exception as e:
            print(f"  {endpoint:35} [FAILED] {str(e)[:20]}")
    
    print(f"\nResult: {working_endpoints}/{len(critical_endpoints)} working, {chart_ready_endpoints} chart-ready")
    
    # 2. TEST DATA STRUCTURES
    print("\n2. TESTING DATA STRUCTURES FOR VISUALIZATION")
    print("-" * 40)
    
    # Check a key endpoint for visualization data
    try:
        response = requests.get(f"{base_url}/api/analytics/summary", timeout=3)
        if response.status_code == 200:
            data = response.json()
            
            print("Analytics Summary Data Structure:")
            print(f"  Has status field: {'YES' if 'status' in data else 'NO'}")
            print(f"  Has timestamp: {'YES' if 'timestamp' in data else 'NO'}")
            print(f"  Has charts object: {'YES' if 'charts' in data else 'NO'}")
            
            if 'charts' in data:
                for chart_name, chart_data in list(data['charts'].items())[:2]:
                    print(f"  Chart '{chart_name}':")
                    if isinstance(chart_data, list) and chart_data:
                        print(f"    - Type: Array with {len(chart_data)} items")
                        if isinstance(chart_data[0], dict):
                            print(f"    - Fields: {list(chart_data[0].keys())[:3]}")
                    elif isinstance(chart_data, dict):
                        print(f"    - Type: Object with {len(chart_data)} properties")
    except Exception as e:
        print(f"  Error checking data structure: {e}")
    
    # 3. TEST REAL-TIME CAPABILITIES
    print("\n3. TESTING REAL-TIME UPDATE CAPABILITIES")
    print("-" * 40)
    
    realtime_endpoint = '/api/analytics/recent'
    print(f"Testing {realtime_endpoint} for real-time updates...")
    
    timestamps = []
    for i in range(3):
        try:
            response = requests.get(f"{base_url}{realtime_endpoint}", timeout=3)
            if response.status_code == 200:
                data = response.json()
                if 'timestamp' in data:
                    timestamps.append(data['timestamp'])
                time.sleep(1)
        except:
            break
    
    if len(timestamps) >= 2:
        unique_timestamps = len(set(timestamps))
        print(f"  Collected {len(timestamps)} responses")
        print(f"  Unique timestamps: {unique_timestamps}")
        print(f"  Real-time updates: {'YES' if unique_timestamps > 1 else 'NO'}")
    else:
        print("  Could not test real-time capabilities")
    
    # 4. TEST CONCURRENT PERFORMANCE
    print("\n4. TESTING PERFORMANCE UNDER CONCURRENT LOAD")
    print("-" * 40)
    print("Simulating 10 concurrent frontend requests...")
    
    def make_request(endpoint):
        try:
            start = time.time()
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            return {
                'success': response.status_code == 200,
                'time': (time.time() - start) * 1000
            }
        except:
            return {'success': False, 'time': 0}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        test_endpoint = '/api/health/live'
        for _ in range(10):
            future = executor.submit(make_request, test_endpoint)
            futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=10):
            try:
                result = future.result()
                results.append(result)
            except:
                pass
    
    successful = sum(1 for r in results if r['success'])
    response_times = [r['time'] for r in results if r['success']]
    
    print(f"  Requests: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Success rate: {(successful/len(results)*100):.1f}%")
    if response_times:
        print(f"  Avg response time: {sum(response_times)/len(response_times):.1f}ms")
    
    # 5. TEST ERROR HANDLING
    print("\n5. TESTING ERROR HANDLING FOR FRONTEND")
    print("-" * 40)
    
    error_endpoint = '/api/nonexistent'
    try:
        response = requests.get(f"{base_url}{error_endpoint}", timeout=3)
        print(f"  Invalid endpoint test:")
        print(f"    Status code: {response.status_code}")
        print(f"    Proper error (4xx/5xx): {'YES' if 400 <= response.status_code < 600 else 'NO'}")
        print(f"    Has error message: {'YES' if response.text else 'NO'}")
    except Exception as e:
        print(f"  Error test failed: {e}")
    
    # FINAL SUMMARY
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    # Calculate overall score
    scores = []
    scores.append((working_endpoints / len(critical_endpoints)) * 100)  # API availability
    scores.append((chart_ready_endpoints / len(critical_endpoints)) * 100)  # Chart readiness
    scores.append(100 if len(set(timestamps)) > 1 else 50)  # Real-time capability
    scores.append((successful / 10) * 100 if results else 0)  # Performance
    
    overall_score = sum(scores) / len(scores)
    
    print(f"\nSCORES:")
    print(f"  API Availability:    {scores[0]:.1f}%")
    print(f"  Chart Readiness:     {scores[1]:.1f}%")
    print(f"  Real-time Updates:   {scores[2]:.1f}%")
    print(f"  Performance:         {scores[3]:.1f}%")
    print(f"  OVERALL:             {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("\nSTATUS: EXCELLENT - Frontend integration is production ready!")
        print("All backend capabilities are properly exposed for frontend.")
    elif overall_score >= 60:
        print("\nSTATUS: GOOD - Frontend integration is working well.")
        print("Minor improvements could enhance the experience.")
    else:
        print("\nSTATUS: NEEDS IMPROVEMENT - Some integration issues exist.")
        print("Review endpoints and data structures for frontend compatibility.")
    
    print("\nKEY FINDINGS:")
    print(f"- {working_endpoints} of {len(critical_endpoints)} critical endpoints are working")
    print(f"- {chart_ready_endpoints} endpoints provide chart-ready data")
    print(f"- Real-time updates are {'working' if len(set(timestamps)) > 1 else 'not detected'}")
    print(f"- System handles concurrent requests with {(successful/10*100):.0f}% success rate")
    
    print("\nThis automated test validates frontend integration without any browser!")
    
    return overall_score >= 60


if __name__ == "__main__":
    success = test_frontend_integration()
    exit(0 if success else 1)