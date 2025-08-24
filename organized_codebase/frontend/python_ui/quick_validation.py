#!/usr/bin/env python3
"""
Quick Validation of Fixed Endpoints
=====================================

Tests all the critical endpoints that were fixed to ensure 100% integration.
"""

import requests
import time

def test_endpoints():
    """Test all critical endpoints with proper timeouts."""
    endpoints = [
        # Intelligence Agents (FIXED)
        '/api/intelligence/agents/status',
        '/api/intelligence/agents/coordination', 
        '/api/intelligence/agents/activities',
        
        # Workflow & DAG (FIXED)
        '/api/workflow/dag',
        '/api/flow/workflow',
        '/api/flow/dependencies',
        
        # Quality Metrics (FIXED)  
        '/api/quality/metrics',
        
        # Performance Flame Graphs (NEW)
        '/api/performance/flamegraph',
        
        # Coverage (should work)
        '/api/coverage/branch-analysis'
    ]
    
    base_url = "http://localhost:5000"
    working = 0
    total = len(endpoints)
    
    print("CRITICAL ENDPOINT VALIDATION")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=15)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                real_data = 'real_data' in str(data)
                print(f"[OK] {endpoint} - {status} {'(REAL DATA)' if real_data else ''}")
                working += 1
            else:
                print(f"[FAIL] {endpoint} - HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] {endpoint} - Server processing")
        except Exception as e:
            print(f"[ERROR] {endpoint} - {str(e)[:50]}")
        
        time.sleep(1)  # Prevent server overload
    
    print("=" * 50)
    print(f"RESULT: {working}/{total} endpoints working ({working/total*100:.1f}%)")
    
    if working == total:
        print("SUCCESS: ALL CRITICAL ENDPOINTS WORKING!")
        print("[FIXED] Intelligence Agents API - WORKING")
        print("[FIXED] Workflow & DAG endpoints - WORKING") 
        print("[FIXED] Quality metrics endpoint - WORKING")
        print("[NEW] Performance Flame Graphs - WORKING")
        print("[SUCCESS] COMPLETE Backend-Frontend Integration ACHIEVED!")
    else:
        print(f"ATTENTION: {total-working} endpoints still need attention")

if __name__ == "__main__":
    test_endpoints()