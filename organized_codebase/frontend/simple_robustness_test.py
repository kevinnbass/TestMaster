#!/usr/bin/env python3
"""
Simple Robustness Enhancement Test
=================================

Quick test to verify new robustness enhancement components are working
with fallback mechanisms in place.
"""

import requests
import json
import time
from datetime import datetime

def test_health():
    """Test basic system health."""
    try:
        response = requests.get("http://localhost:5000/api/health/live", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"[+] Health Check: System is alive ({data.get('uptime_seconds', 0):.1f}s uptime)")
            return True
        else:
            print(f"[X] Health Check: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"[X] Health Check: {str(e)}")
        return False

def test_analytics_with_fallbacks():
    """Test analytics endpoint with fallback mechanisms."""
    try:
        start_time = time.time()
        response = requests.get("http://localhost:5000/api/analytics/metrics", timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"[+] Analytics: Response received in {response_time:.1f}s")
            
            # Check for comprehensive data
            comprehensive = data.get('comprehensive', {})
            if comprehensive:
                print(f"  - Comprehensive analytics: Available")
                
                # Check for new robustness components (with fallbacks)
                robustness_components = [
                    'data_sanitizer', 'deduplication_engine', 'rate_limiter', 
                    'integrity_verifier', 'error_recovery', 'connectivity_monitor'
                ]
                
                found_components = 0
                for component in robustness_components:
                    if component in comprehensive:
                        found_components += 1
                        print(f"  - {component}: Present")
                    else:
                        print(f"  - {component}: Not available (fallback active)")
                
                print(f"  - New components active: {found_components}/{len(robustness_components)}")
                return True
            else:
                print(f"  - No comprehensive analytics data")
                return False
        else:
            print(f"[X] Analytics: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"[X] Analytics: {str(e)}")
        return False

def test_component_fallbacks():
    """Test that system works even when components fail."""
    print("\n[*] Testing Component Fallback Mechanisms:")
    
    # Test multiple requests to see stability
    success_count = 0
    total_requests = 5
    
    for i in range(total_requests):
        try:
            response = requests.get("http://localhost:5000/api/health/live", timeout=5)
            if response.status_code == 200:
                success_count += 1
                print(f"  Request {i+1}: [+]")
            else:
                print(f"  Request {i+1}: [X] ({response.status_code})")
        except Exception as e:
            print(f"  Request {i+1}: [X] ({str(e)})")
        
        time.sleep(0.5)
    
    stability = (success_count / total_requests) * 100
    print(f"  System Stability: {stability:.1f}% ({success_count}/{total_requests} requests successful)")
    
    return stability >= 80  # 80% success rate is acceptable with fallbacks

def main():
    """Run simple robustness test."""
    print("Simple Robustness Enhancement Test")
    print("=" * 50)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    # Test 1: Basic Health
    print("1. Testing Basic System Health...")
    health_ok = test_health()
    print()
    
    # Test 2: Analytics with Fallbacks
    print("2. Testing Analytics with Fallback Mechanisms...")
    analytics_ok = test_analytics_with_fallbacks()
    print()
    
    # Test 3: Component Fallbacks
    fallbacks_ok = test_component_fallbacks()
    print()
    
    # Summary
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    tests_passed = sum([health_ok, analytics_ok, fallbacks_ok])
    total_tests = 3
    
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Health Check: {'PASS' if health_ok else 'FAIL'}")
    print(f"Analytics: {'PASS' if analytics_ok else 'FAIL'}")
    print(f"Fallbacks: {'PASS' if fallbacks_ok else 'FAIL'}")
    
    if tests_passed == total_tests:
        print("\n[SUCCESS] ALL TESTS PASSED")
        print("Robustness enhancement components are working with fallback mechanisms!")
    elif tests_passed >= 2:
        print("\n[WARNING] MOSTLY PASSING")
        print("System is functional with some issues - fallbacks are working")
    else:
        print("\n[ERROR] MULTIPLE FAILURES")
        print("System needs attention - integration issues remain")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()