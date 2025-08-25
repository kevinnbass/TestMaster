#!/usr/bin/env python3
"""
Test New Visualization APIs
============================

Tests all the newly created visualization endpoints.

Author: TestMaster Team
"""

import requests
import json
from datetime import datetime

def test_new_visualization_apis():
    """Test all new visualization API endpoints."""
    print("=" * 70)
    print("TESTING NEW VISUALIZATION APIs")
    print("=" * 70)
    print()
    
    base_url = "http://localhost:5000"
    
    # Define all new endpoints to test
    endpoints = [
        # Intelligence API
        ('/api/intelligence/agents/status', 'Multi-Agent Status'),
        ('/api/intelligence/agents/coordination', 'Agent Coordination Patterns'),
        ('/api/intelligence/agents/activities', 'Agent Activities'),
        ('/api/intelligence/agents/decisions', 'Consensus Decisions'),
        ('/api/intelligence/agents/optimization', 'Optimization Metrics'),
        
        # Test Generation API
        ('/api/test-generation/generators/status', 'Test Generators Status'),
        ('/api/test-generation/generation/live', 'Live Test Generation'),
        ('/api/test-generation/generation/queue', 'Generation Queue'),
        ('/api/test-generation/generation/performance', 'Generation Performance'),
        ('/api/test-generation/generation/insights', 'Generation Insights'),
        
        # Security API
        ('/api/security/vulnerabilities/heatmap', 'Vulnerability Heatmap'),
        ('/api/security/owasp/compliance', 'OWASP Compliance'),
        ('/api/security/threats/realtime', 'Real-time Threats'),
        ('/api/security/scanning/status', 'Security Scanning Status'),
        ('/api/security/remediation/recommendations', 'Remediation Recommendations')
    ]
    
    # Test each endpoint
    successful = 0
    failed = 0
    chart_ready = 0
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for required fields
                has_status = 'status' in data and data['status'] == 'success'
                has_timestamp = 'timestamp' in data
                has_charts = 'charts' in data
                
                # Calculate data richness
                data_size = len(response.text)
                field_count = len(data) if isinstance(data, dict) else 0
                
                if has_charts:
                    chart_ready += 1
                
                print(f"[OK] {name:40}")
                print(f"     Endpoint: {endpoint}")
                print(f"     Status: {'OK' if has_status else 'MISSING'}")
                print(f"     Timestamp: {'YES' if has_timestamp else 'NO'}")
                print(f"     Charts: {'YES' if has_charts else 'NO'}")
                print(f"     Data size: {data_size} bytes")
                print(f"     Fields: {field_count}")
                
                if has_charts and isinstance(data['charts'], dict):
                    chart_types = list(data['charts'].keys())[:3]
                    print(f"     Chart types: {', '.join(chart_types)}")
                
                print()
                successful += 1
                
            else:
                print(f"[X] {name:40}")
                print(f"    Endpoint: {endpoint}")
                print(f"    Error: HTTP {response.status_code}")
                print()
                failed += 1
                
        except Exception as e:
            print(f"[X] {name:40}")
            print(f"    Endpoint: {endpoint}")
            print(f"    Error: {str(e)[:50]}")
            print()
            failed += 1
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total endpoints tested: {len(endpoints)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Chart-ready: {chart_ready}")
    print(f"Success rate: {(successful/len(endpoints)*100):.1f}%")
    print(f"Chart readiness: {(chart_ready/len(endpoints)*100):.1f}%")
    
    if successful == len(endpoints):
        print("\n[SUCCESS] All new visualization APIs are working perfectly!")
        print("Frontend can now consume:")
        print("  - Multi-Agent Intelligence Dashboard data")
        print("  - Real-time Test Generation monitoring")
        print("  - Security Vulnerability Heatmaps")
        print("  - OWASP Compliance tracking")
        print("  - And much more!")
    elif successful > 0:
        print(f"\n[PARTIAL] {successful}/{len(endpoints)} APIs working")
        print("Some endpoints may need attention")
    else:
        print("\n[FAILED] No APIs are responding")
        print("Check if the server is running on port 5000")
    
    return successful == len(endpoints)


if __name__ == "__main__":
    success = test_new_visualization_apis()
    exit(0 if success else 1)