#!/usr/bin/env python3
"""
Automated Frontend-Backend Integration Tester
==============================================

This tool automatically tests that all backend capabilities are properly 
exposed and working for frontend consumption WITHOUT requiring a web browser.

It simulates everything a frontend would do:
- API calls
- Data validation
- Chart data structure verification
- Real-time updates
- Error handling
- Performance testing

Author: TestMaster Team
"""

import requests
import json
import time
import asyncio
import websockets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import concurrent.futures
import statistics
import hashlib
import random

class AutomatedFrontendTester:
    """Comprehensive automated frontend testing without browser."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = {
            'api_endpoints': {},
            'data_structures': {},
            'real_time_features': {},
            'performance_metrics': {},
            'error_handling': {},
            'overall_health': {}
        }
        
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run complete frontend integration test suite."""
        print("=" * 70)
        print("AUTOMATED FRONTEND-BACKEND INTEGRATION TEST")
        print("Testing without browser intervention...")
        print("=" * 70)
        print()
        
        # 1. Test API Endpoint Availability
        print("1. TESTING API ENDPOINTS")
        print("-" * 40)
        self.test_api_endpoints()
        
        # 2. Test Data Structure Quality
        print("\n2. TESTING DATA STRUCTURES FOR FRONTEND")
        print("-" * 40)
        self.test_data_structures()
        
        # 3. Test Real-time Features
        print("\n3. TESTING REAL-TIME FEATURES")
        print("-" * 40)
        self.test_realtime_features()
        
        # 4. Test Performance Under Load
        print("\n4. TESTING PERFORMANCE UNDER LOAD")
        print("-" * 40)
        self.test_performance()
        
        # 5. Test Error Handling
        print("\n5. TESTING ERROR HANDLING")
        print("-" * 40)
        self.test_error_handling()
        
        # 6. Test WebSocket Connections (if available)
        print("\n6. TESTING WEBSOCKET CONNECTIONS")
        print("-" * 40)
        self.test_websockets()
        
        # 7. Generate Overall Health Report
        print("\n7. GENERATING HEALTH REPORT")
        print("-" * 40)
        self.generate_health_report()
        
        return self.test_results
    
    def test_api_endpoints(self):
        """Test all API endpoints for availability and response quality."""
        endpoints = {
            # Health & Monitoring
            '/api/health/live': {'category': 'health', 'critical': True},
            '/api/health/ready': {'category': 'health', 'critical': True},
            
            # Ultra-Reliability Features
            '/api/monitoring/robustness': {'category': 'reliability', 'critical': True},
            '/api/monitoring/heartbeat': {'category': 'reliability', 'critical': False},
            '/api/monitoring/fallback': {'category': 'reliability', 'critical': False},
            '/api/monitoring/dead-letter': {'category': 'reliability', 'critical': False},
            '/api/monitoring/batch': {'category': 'reliability', 'critical': False},
            '/api/monitoring/flow': {'category': 'reliability', 'critical': False},
            '/api/monitoring/compression': {'category': 'reliability', 'critical': False},
            
            # Analytics
            '/api/analytics/summary': {'category': 'analytics', 'critical': True},
            '/api/analytics/recent': {'category': 'analytics', 'critical': True},
            '/api/analytics/trends': {'category': 'analytics', 'critical': False},
            '/api/analytics/export': {'category': 'analytics', 'critical': False},
            '/api/analytics/insights': {'category': 'analytics', 'critical': False},
            
            # Performance
            '/api/performance/metrics': {'category': 'performance', 'critical': True},
            '/api/performance/summary': {'category': 'performance', 'critical': False},
            '/api/performance/history': {'category': 'performance', 'critical': False}
        }
        
        results = {}
        for endpoint, config in endpoints.items():
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Validate data quality
                    quality_score = self._calculate_data_quality(data)
                    
                    results[endpoint] = {
                        'status': 'SUCCESS',
                        'response_time_ms': response_time,
                        'data_size': len(response.text),
                        'quality_score': quality_score,
                        'has_charts': 'charts' in data,
                        'has_timestamp': 'timestamp' in data,
                        'category': config['category'],
                        'critical': config['critical']
                    }
                    
                    status_icon = "[OK]" if quality_score >= 70 else "[!]"
                    print(f"{status_icon} {endpoint:35} OK ({response_time:.1f}ms, Quality: {quality_score}%)")
                else:
                    results[endpoint] = {
                        'status': 'ERROR',
                        'status_code': response.status_code,
                        'category': config['category'],
                        'critical': config['critical']
                    }
                    print(f"[X] {endpoint:35} ERROR ({response.status_code})")
                    
            except Exception as e:
                results[endpoint] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'category': config['category'],
                    'critical': config['critical']
                }
                print(f"[X] {endpoint:35} FAILED ({str(e)[:30]})")
        
        self.test_results['api_endpoints'] = results
        
        # Summary
        total = len(results)
        successful = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        critical_ok = all(r['status'] == 'SUCCESS' for endpoint, r in results.items() 
                         if r.get('critical', False))
        
        print(f"\nEndpoint Summary: {successful}/{total} working")
        print(f"Critical Endpoints: {'ALL OK' if critical_ok else 'SOME FAILED'}")
    
    def test_data_structures(self):
        """Test data structures for frontend consumption."""
        test_endpoints = [
            '/api/analytics/summary',
            '/api/performance/metrics',
            '/api/monitoring/robustness'
        ]
        
        results = {}
        for endpoint in test_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Analyze data structure
                    structure_analysis = {
                        'is_valid_json': True,
                        'has_status_field': 'status' in data,
                        'has_timestamp': 'timestamp' in data,
                        'has_charts': 'charts' in data,
                        'chart_types': list(data.get('charts', {}).keys()) if 'charts' in data else [],
                        'data_depth': self._calculate_data_depth(data),
                        'field_count': len(data),
                        'has_arrays': any(isinstance(v, list) for v in data.values()),
                        'has_nested_objects': any(isinstance(v, dict) for v in data.values())
                    }
                    
                    # Calculate structure score
                    score = sum([
                        structure_analysis['has_status_field'] * 20,
                        structure_analysis['has_timestamp'] * 20,
                        structure_analysis['has_charts'] * 30,
                        (structure_analysis['data_depth'] >= 2) * 15,
                        (structure_analysis['field_count'] >= 3) * 15
                    ])
                    
                    structure_analysis['structure_score'] = score
                    results[endpoint] = structure_analysis
                    
                    print(f"[OK] {endpoint:35} Score: {score}%")
                    if structure_analysis['chart_types']:
                        print(f"  Charts available: {', '.join(structure_analysis['chart_types'][:3])}")
                else:
                    results[endpoint] = {'status': 'ERROR', 'status_code': response.status_code}
                    print(f"[X] {endpoint:35} ERROR")
                    
            except Exception as e:
                results[endpoint] = {'status': 'FAILED', 'error': str(e)}
                print(f"[X] {endpoint:35} FAILED")
        
        self.test_results['data_structures'] = results
    
    def test_realtime_features(self):
        """Test real-time data update capabilities."""
        realtime_endpoints = [
            '/api/analytics/recent',
            '/api/health/live',
            '/api/monitoring/heartbeat'
        ]
        
        results = {}
        for endpoint in realtime_endpoints:
            try:
                # Make multiple requests to check for data changes
                responses = []
                for i in range(3):
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=3)
                    if response.status_code == 200:
                        responses.append({
                            'timestamp': time.time(),
                            'data': response.json()
                        })
                        time.sleep(1)
                
                if len(responses) >= 2:
                    # Check for timestamp changes
                    timestamps_differ = len(set(
                        r['data'].get('timestamp', '') for r in responses
                    )) > 1
                    
                    # Check for data variations
                    data_hashes = [
                        hashlib.md5(json.dumps(r['data'], sort_keys=True).encode()).hexdigest()
                        for r in responses
                    ]
                    data_varies = len(set(data_hashes)) > 1
                    
                    results[endpoint] = {
                        'realtime_capable': timestamps_differ or data_varies,
                        'timestamps_update': timestamps_differ,
                        'data_changes': data_varies,
                        'response_count': len(responses)
                    }
                    
                    status = "[OK]" if results[endpoint]['realtime_capable'] else "!"
                    print(f"{status} {endpoint:35} Real-time: {results[endpoint]['realtime_capable']}")
                else:
                    results[endpoint] = {'status': 'INCOMPLETE'}
                    print(f"! {endpoint:35} INCOMPLETE")
                    
            except Exception as e:
                results[endpoint] = {'status': 'FAILED', 'error': str(e)}
                print(f"[X] {endpoint:35} FAILED")
        
        self.test_results['real_time_features'] = results
    
    def test_performance(self):
        """Test performance under concurrent load."""
        print("Simulating concurrent frontend requests...")
        
        test_endpoints = [
            '/api/health/live',
            '/api/analytics/summary',
            '/api/performance/metrics'
        ]
        
        def make_request(endpoint):
            try:
                start = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                return {
                    'endpoint': endpoint,
                    'success': response.status_code == 200,
                    'response_time': (time.time() - start) * 1000,
                    'size': len(response.text) if response.status_code == 200 else 0
                }
            except Exception as e:
                return {
                    'endpoint': endpoint,
                    'success': False,
                    'response_time': 0,
                    'error': str(e)
                }
        
        # Simulate 30 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(10):
                for endpoint in test_endpoints:
                    future = executor.submit(make_request, endpoint)
                    futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e)})
        
        # Analyze performance
        successful = [r for r in results if r.get('success', False)]
        response_times = [r['response_time'] for r in successful]
        
        performance_metrics = {
            'total_requests': len(results),
            'successful_requests': len(successful),
            'success_rate': (len(successful) / len(results) * 100) if results else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'p95_response_time': sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0
        }
        
        self.test_results['performance_metrics'] = performance_metrics
        
        print(f"Success Rate: {performance_metrics['success_rate']:.1f}%")
        print(f"Avg Response: {performance_metrics['avg_response_time']:.1f}ms")
        print(f"P95 Response: {performance_metrics['p95_response_time']:.1f}ms")
    
    def test_error_handling(self):
        """Test error handling for invalid requests."""
        error_tests = [
            ('/api/nonexistent', 404, 'Nonexistent endpoint'),
            ('/api/analytics/invalid-id', 404, 'Invalid resource'),
            ('/api/health/unauthorized', 404, 'Unauthorized access')
        ]
        
        results = {}
        for endpoint, expected_status, description in error_tests:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=3)
                
                proper_error = 400 <= response.status_code < 600
                has_error_message = len(response.text) > 0
                
                results[endpoint] = {
                    'description': description,
                    'status_code': response.status_code,
                    'expected_status': expected_status,
                    'proper_error_handling': proper_error,
                    'has_error_message': has_error_message,
                    'passed': proper_error
                }
                
                status = "[OK]" if proper_error else "[X]"
                print(f"{status} {description:35} Status: {response.status_code}")
                
            except Exception as e:
                results[endpoint] = {
                    'description': description,
                    'error': str(e),
                    'passed': False
                }
                print(f"[X] {description:35} Exception")
        
        self.test_results['error_handling'] = results
    
    def test_websockets(self):
        """Test WebSocket connections for real-time features."""
        websocket_endpoints = [
            'ws://localhost:8765',  # Health monitor WebSocket
            'ws://localhost:8766'   # Analytics WebSocket
        ]
        
        results = {}
        
        async def test_ws_connection(url):
            try:
                async with websockets.connect(url, timeout=3) as websocket:
                    # Send test message
                    test_msg = json.dumps({
                        'type': 'test',
                        'timestamp': datetime.now().isoformat()
                    })
                    await websocket.send(test_msg)
                    
                    # Try to receive response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2)
                        return {
                            'connected': True,
                            'responsive': True,
                            'response_received': True
                        }
                    except asyncio.TimeoutError:
                        return {
                            'connected': True,
                            'responsive': False,
                            'response_received': False
                        }
            except Exception as e:
                return {
                    'connected': False,
                    'error': str(e)
                }
        
        # Test each WebSocket endpoint
        for url in websocket_endpoints:
            try:
                result = asyncio.run(test_ws_connection(url))
                results[url] = result
                
                if result['connected']:
                    status = "[OK]" if result.get('responsive', False) else "!"
                    print(f"{status} {url:35} Connected: {result['connected']}")
                else:
                    print(f"[X] {url:35} Not available")
                    
            except Exception as e:
                results[url] = {'error': str(e)}
                print(f"[X] {url:35} Error")
        
        self.test_results['websockets'] = results
    
    def generate_health_report(self):
        """Generate comprehensive health report."""
        
        # Calculate overall health score
        api_health = self._calculate_api_health()
        data_health = self._calculate_data_health()
        realtime_health = self._calculate_realtime_health()
        performance_health = self._calculate_performance_health()
        error_health = self._calculate_error_health()
        
        overall_health = {
            'api_health': api_health,
            'data_structure_health': data_health,
            'realtime_capability': realtime_health,
            'performance_score': performance_health,
            'error_handling_score': error_health,
            'overall_score': statistics.mean([
                api_health, data_health, realtime_health, 
                performance_health, error_health
            ])
        }
        
        self.test_results['overall_health'] = overall_health
        
        # Print summary
        print(f"\nHEALTH SCORES:")
        print(f"  API Health:        {api_health:.1f}%")
        print(f"  Data Structures:   {data_health:.1f}%")
        print(f"  Real-time:         {realtime_health:.1f}%")
        print(f"  Performance:       {performance_health:.1f}%")
        print(f"  Error Handling:    {error_health:.1f}%")
        print(f"  OVERALL:           {overall_health['overall_score']:.1f}%")
        
        # Determine status
        score = overall_health['overall_score']
        if score >= 90:
            status = "EXCELLENT - Production Ready"
        elif score >= 75:
            status = "GOOD - Minor improvements needed"
        elif score >= 60:
            status = "FAIR - Some issues to address"
        else:
            status = "POOR - Significant improvements required"
        
        print(f"\nSTATUS: {status}")
        
        # Save detailed report
        self._save_report()
    
    def _calculate_data_quality(self, data: Any) -> int:
        """Calculate data quality score for frontend consumption."""
        if not isinstance(data, dict):
            return 0
        
        score = 0
        
        # Basic structure (40 points)
        if 'status' in data:
            score += 20
        if 'timestamp' in data:
            score += 20
        
        # Rich data (30 points)
        if 'charts' in data and isinstance(data['charts'], dict):
            score += 30
        elif len(data) >= 5:
            score += 20
        elif len(data) >= 3:
            score += 10
        
        # Data depth (30 points)
        depth = self._calculate_data_depth(data)
        if depth >= 3:
            score += 30
        elif depth >= 2:
            score += 20
        elif depth >= 1:
            score += 10
        
        return min(score, 100)
    
    def _calculate_data_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested data structures."""
        if current_depth > 5:
            return current_depth
        
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._calculate_data_depth(v, current_depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._calculate_data_depth(item, current_depth + 1) for item in data[:5])
        else:
            return current_depth
    
    def _calculate_api_health(self) -> float:
        """Calculate API endpoint health score."""
        endpoints = self.test_results.get('api_endpoints', {})
        if not endpoints:
            return 0
        
        total = len(endpoints)
        successful = sum(1 for e in endpoints.values() if e.get('status') == 'SUCCESS')
        critical_ok = all(e.get('status') == 'SUCCESS' for e in endpoints.values() 
                         if e.get('critical', False))
        
        base_score = (successful / total) * 80
        critical_bonus = 20 if critical_ok else 0
        
        return base_score + critical_bonus
    
    def _calculate_data_health(self) -> float:
        """Calculate data structure health score."""
        structures = self.test_results.get('data_structures', {})
        if not structures:
            return 0
        
        scores = [s.get('structure_score', 0) for s in structures.values() 
                 if isinstance(s, dict) and 'structure_score' in s]
        
        return statistics.mean(scores) if scores else 0
    
    def _calculate_realtime_health(self) -> float:
        """Calculate real-time capability score."""
        realtime = self.test_results.get('real_time_features', {})
        if not realtime:
            return 0
        
        capable = sum(1 for r in realtime.values() 
                     if r.get('realtime_capable', False))
        total = len(realtime)
        
        return (capable / total * 100) if total > 0 else 0
    
    def _calculate_performance_health(self) -> float:
        """Calculate performance health score."""
        perf = self.test_results.get('performance_metrics', {})
        if not perf:
            return 0
        
        score = 100
        
        # Deduct points for poor performance
        if perf.get('success_rate', 0) < 95:
            score -= 20
        if perf.get('avg_response_time', 0) > 1000:
            score -= 20
        if perf.get('p95_response_time', 0) > 2000:
            score -= 10
        
        return max(score, 0)
    
    def _calculate_error_health(self) -> float:
        """Calculate error handling health score."""
        errors = self.test_results.get('error_handling', {})
        if not errors:
            return 0
        
        passed = sum(1 for e in errors.values() if e.get('passed', False))
        total = len(errors)
        
        return (passed / total * 100) if total > 0 else 0
    
    def _save_report(self):
        """Save detailed test report to file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'summary': {
                'overall_score': self.test_results['overall_health']['overall_score'],
                'api_endpoints_tested': len(self.test_results.get('api_endpoints', {})),
                'data_structures_validated': len(self.test_results.get('data_structures', {})),
                'realtime_features_tested': len(self.test_results.get('real_time_features', {})),
                'performance_metrics': self.test_results.get('performance_metrics', {}),
                'error_tests_passed': sum(1 for e in self.test_results.get('error_handling', {}).values() 
                                        if e.get('passed', False))
            }
        }
        
        with open('frontend_integration_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\nDetailed report saved to: frontend_integration_test_report.json")


def main():
    """Run automated frontend testing."""
    tester = AutomatedFrontendTester()
    results = tester.run_complete_test_suite()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    overall_score = results['overall_health']['overall_score']
    if overall_score >= 80:
        print(f"\n[OK] Frontend integration is READY ({overall_score:.1f}%)")
        print("All backend capabilities are properly exposed for frontend consumption.")
    else:
        print(f"\n! Frontend integration needs improvement ({overall_score:.1f}%)")
        print("Review the detailed report for specific issues to address.")
    
    return overall_score >= 80


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)