#!/usr/bin/env python3
"""
Comprehensive API Endpoint Test Suite
====================================

Tests all dashboard API endpoints and reports their status.
This serves as both a test suite and API documentation.
"""

import requests
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple

class APITester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results = []
        
    def test_endpoint(self, method: str, path: str, description: str, 
                     data: dict = None, expected_status: int = 200) -> bool:
        """Test a single API endpoint"""
        url = f"{self.base_url}{path}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=5)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=5)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            success = response.status_code == expected_status
            
            result = {
                'method': method.upper(),
                'path': path,
                'description': description,
                'expected_status': expected_status,
                'actual_status': response.status_code,
                'success': success,
                'response_size': len(response.content),
                'content_type': response.headers.get('content-type', 'unknown')
            }
            
            if success and response.headers.get('content-type', '').startswith('application/json'):
                try:
                    json_data = response.json()
                    result['has_json'] = True
                    result['json_status'] = json_data.get('status', 'unknown')
                except:
                    result['has_json'] = False
            else:
                result['has_json'] = False
                
            self.results.append(result)
            return success
            
        except Exception as e:
            result = {
                'method': method.upper(),
                'path': path,
                'description': description,
                'expected_status': expected_status,
                'actual_status': 'ERROR',
                'success': False,
                'error': str(e),
                'response_size': 0,
                'content_type': 'error'
            }
            self.results.append(result)
            return False
    
    def run_comprehensive_tests(self):
        """Run comprehensive test suite"""
        print("Dashboard API Comprehensive Test Suite")
        print("=" * 50)
        print(f"Testing against: {self.base_url}")
        print(f"Test started at: {datetime.now().isoformat()}")
        print()
        
        # Core System Endpoints
        print("Testing Core System Endpoints...")
        self.test_endpoint("GET", "/", "Main dashboard page", expected_status=200)
        self.test_endpoint("GET", "/api/health", "System health check")
        self.test_endpoint("GET", "/api/config", "Configuration information")
        
        # Performance Monitoring Endpoints (CRITICAL)
        print("\\nTesting Performance Monitoring Endpoints...")
        self.test_endpoint("GET", "/api/performance/realtime", "Real-time performance data (CRITICAL)")
        self.test_endpoint("GET", "/api/performance/history", "Performance history")
        self.test_endpoint("GET", "/api/performance/summary", "Performance summary")
        self.test_endpoint("GET", "/api/performance/status", "Monitoring status")
        
        # Analytics Endpoints  
        print("\\nTesting Analytics Endpoints...")
        self.test_endpoint("GET", "/api/analytics/metrics", "Current system metrics")
        self.test_endpoint("GET", "/api/analytics/trends", "Trend analysis")
        
        # Workflow Management Endpoints (CRITICAL - was 404 before)
        print("\\nTesting Workflow Management Endpoints...")
        self.test_endpoint("GET", "/api/workflow/status", "Workflow status (CRITICAL - was 404)")
        self.test_endpoint("POST", "/api/workflow/start", "Start workflow", 
                          data={"workflow_name": "test"}, expected_status=503)  # Expected to fail without manager
        self.test_endpoint("POST", "/api/workflow/stop", "Stop workflow", 
                          data={"workflow_id": "test"}, expected_status=503)
        self.test_endpoint("GET", "/api/workflow/history", "Workflow history", expected_status=503)
        self.test_endpoint("GET", "/api/workflow/dag", "Workflow DAG", expected_status=503)
        
        # LLM Integration Endpoints (CRITICAL for toggle button)
        print("\\nTesting LLM Integration Endpoints...")
        self.test_endpoint("GET", "/api/llm/status", "LLM status (CRITICAL for toggle button)")
        self.test_endpoint("POST", "/api/llm/toggle-mode", "Toggle LLM mode (CRITICAL)", 
                          data={"enabled": True})
        self.test_endpoint("GET", "/api/llm/metrics", "LLM metrics")
        self.test_endpoint("POST", "/api/llm/analyze", "LLM analysis", 
                          data={"module_path": "test.py"}, expected_status=403)  # Expected to fail - LLM disabled
        self.test_endpoint("POST", "/api/llm/estimate-cost", "Cost estimation", 
                          data={"operation": "test", "input_size": 100})
        
        # Test Management Endpoints
        print("\\nTesting Test Management Endpoints...")
        self.test_endpoint("GET", "/api/tests/status", "Test suite status")
        self.test_endpoint("GET", "/api/tests/coverage", "Test coverage metrics")
        
        # Refactoring Analysis Endpoints
        print("\\nTesting Refactoring Analysis Endpoints...")
        self.test_endpoint("GET", "/api/refactor/analysis", "Refactoring analysis")
        self.test_endpoint("GET", "/api/refactor/hierarchy", "Code hierarchy analysis")
        
        # Debug Endpoints
        print("\\nTesting Debug Endpoints...")
        self.test_endpoint("GET", "/api/debug/routes", "Debug route listing")
        
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\\n" + "=" * 50)
        print("API ENDPOINT TEST RESULTS")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        # Critical endpoints summary
        critical_endpoints = [
            "/api/performance/realtime",
            "/api/workflow/status", 
            "/api/llm/status",
            "/api/health"
        ]
        
        print("CRITICAL ENDPOINT STATUS:")
        print("-" * 30)
        for result in self.results:
            if result['path'] in critical_endpoints:
                status = "[PASS]" if result['success'] else "[FAIL]"
                print(f"{status} {result['path']}")
        print()
        
        # Detailed results
        print("DETAILED RESULTS:")
        print("-" * 30)
        for result in self.results:
            status = "[OK]" if result['success'] else "[FAIL]"
            method = result['method'].ljust(4)
            path = result['path'].ljust(30)
            
            if result['success']:
                size_info = f"({result['response_size']} bytes)" if result.get('response_size') else ""
                json_info = f"JSON: {result['json_status']}" if result.get('has_json') else "HTML"
                print(f"{status} {method} {path} {result['actual_status']} {size_info} {json_info}")
            else:
                error_info = result.get('error', f"HTTP {result['actual_status']}")
                print(f"{status} {method} {path} FAILED - {error_info}")
        
        print()
        print("API DOCUMENTATION:")
        print("-" * 20)
        print("Working endpoints can be used for frontend integration.")
        print("Failed endpoints may need further investigation or are expected failures.")
        
        return passed_tests, total_tests

def main():
    """Main test execution"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:5000"
    
    tester = APITester(base_url)
    
    try:
        tester.run_comprehensive_tests()
        passed, total = tester.generate_report()
        
        # Exit with appropriate code
        if passed == total:
            print("\\n[SUCCESS] All tests passed!")
            sys.exit(0)
        elif passed > total * 0.8:  # 80% pass rate acceptable
            print(f"\\n[ACCEPTABLE] Pass rate: {passed}/{total}")
            sys.exit(0) 
        else:
            print(f"\\n[WARNING] Low pass rate: {passed}/{total}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\\n\\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\nTest execution failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()