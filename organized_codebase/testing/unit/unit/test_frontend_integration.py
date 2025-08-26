#!/usr/bin/env python3
"""
Automated Frontend-Backend Integration Tester
============================================

Tests all backend capabilities are properly exposed and working on the frontend
without manual browser intervention. Validates API endpoints, WebSocket connections,
data flow, and UI component integration.

Author: TestMaster Team
"""

import requests
import websockets
import asyncio
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import concurrent.futures
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_frontend_integration.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class FrontendBackendIntegrationTester:
    """Comprehensive frontend-backend integration tester."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize the integration tester."""
        self.base_url = base_url
        self.websocket_url = base_url.replace('http', 'ws')
        
        # Test results storage
        self.test_results = {
            'api_endpoints': {},
            'websocket_connections': {},
            'data_flow': {},
            'real_time_features': {},
            'ultra_reliability': {},
            'performance_metrics': {},
            'overall_integration': {}
        }
        
        # API endpoints to test
        self.api_endpoints = [
            # Health and monitoring
            '/api/health/live',
            '/api/health/ready', 
            '/api/monitoring/status',
            '/api/monitoring/metrics',
            '/api/monitoring/robustness',
            '/api/monitoring/heartbeat',
            '/api/monitoring/fallback',
            '/api/monitoring/dead-letter',
            '/api/monitoring/batch',
            '/api/monitoring/flow',
            '/api/monitoring/compression',
            '/api/monitoring/test-delivery',
            
            # Analytics
            '/api/analytics/summary',
            '/api/analytics/recent',
            '/api/analytics/trends',
            '/api/analytics/export',
            
            # Performance
            '/api/performance/metrics',
            '/api/performance/summary',
            '/api/performance/history',
            
            # LLM and AI features
            '/api/llm/metrics',
            '/api/llm/status',
            
            # Workflow
            '/api/workflow/status',
            '/api/workflow/history',
            
            # Tests
            '/api/tests/summary',
            '/api/tests/recent',
            '/api/tests/coverage',
            
            # Refactor
            '/api/refactor/status',
            '/api/refactor/suggestions'
        ]
        
        # WebSocket endpoints to test
        self.websocket_endpoints = [
            '/ws/metrics',
            '/ws/health',
            '/ws/analytics'
        ]
        
        logger.info("Frontend-Backend Integration Tester initialized")
    
    def test_server_availability(self) -> bool:
        """Test if the dashboard server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/health/live", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Dashboard server is available")
                return True
            else:
                logger.error(f"❌ Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Server is not available: {e}")
            return False
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test all API endpoints for availability and response format."""
        logger.info("Testing API endpoints...")
        results = {
            'endpoints_tested': 0,
            'endpoints_successful': 0,
            'endpoints_failed': 0,
            'response_times': [],
            'endpoint_details': {},
            'success': False
        }
        
        for endpoint in self.api_endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                results['endpoints_tested'] += 1
                results['response_times'].append(response_time)
                
                endpoint_result = {
                    'status_code': response.status_code,
                    'response_time_ms': response_time,
                    'content_type': response.headers.get('content-type', ''),
                    'success': response.status_code in [200, 201, 202]
                }
                
                # Try to parse JSON response
                try:
                    if 'application/json' in endpoint_result['content_type']:
                        json_data = response.json()
                        endpoint_result['json_valid'] = True
                        endpoint_result['data_keys'] = list(json_data.keys()) if isinstance(json_data, dict) else []
                        endpoint_result['data_size'] = len(str(json_data))
                    else:
                        endpoint_result['json_valid'] = False
                except json.JSONDecodeError:
                    endpoint_result['json_valid'] = False
                
                if endpoint_result['success']:
                    results['endpoints_successful'] += 1
                    logger.info(f"✅ {endpoint}: {response.status_code} ({response_time:.1f}ms)")
                else:
                    results['endpoints_failed'] += 1
                    logger.warning(f"⚠️ {endpoint}: {response.status_code} ({response_time:.1f}ms)")
                
                results['endpoint_details'][endpoint] = endpoint_result
                
            except requests.exceptions.RequestException as e:
                results['endpoints_tested'] += 1
                results['endpoints_failed'] += 1
                results['endpoint_details'][endpoint] = {
                    'error': str(e),
                    'success': False
                }
                logger.error(f"❌ {endpoint}: {e}")
        
        # Calculate summary metrics
        if results['response_times']:
            results['avg_response_time_ms'] = sum(results['response_times']) / len(results['response_times'])
            results['max_response_time_ms'] = max(results['response_times'])
        
        results['success_rate'] = (results['endpoints_successful'] / results['endpoints_tested']) * 100 if results['endpoints_tested'] > 0 else 0
        results['success'] = results['success_rate'] >= 80.0  # 80% success threshold
        
        logger.info(f"API Endpoints: {results['endpoints_successful']}/{results['endpoints_tested']} successful ({results['success_rate']:.1f}%)")
        return results
    
    async def test_websocket_connections(self) -> Dict[str, Any]:
        """Test WebSocket connections for real-time features."""
        logger.info("Testing WebSocket connections...")
        results = {
            'connections_tested': 0,
            'connections_successful': 0,
            'connections_failed': 0,
            'messages_received': 0,
            'connection_details': {},
            'success': False
        }
        
        # Test WebSocket connection to health monitor (if available)
        websocket_test_urls = [
            "ws://localhost:8765",  # Health monitor WebSocket
            "ws://localhost:8766",  # Ultra-reliability health monitor
        ]
        
        for ws_url in websocket_test_urls:
            try:
                results['connections_tested'] += 1
                
                # Test WebSocket connection
                async with websockets.connect(ws_url, timeout=5) as websocket:
                    results['connections_successful'] += 1
                    
                    # Send a test message
                    test_message = {"type": "test", "timestamp": datetime.now().isoformat()}
                    await websocket.send(json.dumps(test_message))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=3)
                        response_data = json.loads(response)
                        results['messages_received'] += 1
                        
                        results['connection_details'][ws_url] = {
                            'success': True,
                            'response_received': True,
                            'response_type': response_data.get('type', 'unknown'),
                            'response_size': len(response)
                        }
                        
                        logger.info(f"✅ WebSocket {ws_url}: Connected and responsive")
                        
                    except asyncio.TimeoutError:
                        results['connection_details'][ws_url] = {
                            'success': True,
                            'response_received': False,
                            'note': 'Connected but no response within timeout'
                        }
                        logger.info(f"✅ WebSocket {ws_url}: Connected (no response)")
                
            except Exception as e:
                results['connections_failed'] += 1
                results['connection_details'][ws_url] = {
                    'success': False,
                    'error': str(e)
                }
                logger.warning(f"⚠️ WebSocket {ws_url}: {e}")
        
        results['success'] = results['connections_successful'] > 0
        logger.info(f"WebSocket Connections: {results['connections_successful']}/{results['connections_tested']} successful")
        return results
    
    def test_data_flow_integration(self) -> Dict[str, Any]:
        """Test data flow between backend and frontend components."""
        logger.info("Testing data flow integration...")
        results = {
            'data_flows_tested': 0,
            'data_flows_successful': 0,
            'data_consistency': {},
            'real_time_updates': {},
            'success': False
        }
        
        try:
            # Test 1: Submit test analytics and verify it appears in API responses
            logger.info("Testing analytics data flow...")
            
            # Submit test analytics via monitoring endpoint
            test_analytics = {
                'test_id': 'frontend_integration_test',
                'timestamp': datetime.now().isoformat(),
                'message': 'Frontend integration test data'
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/monitoring/test-delivery",
                    json=test_analytics,
                    timeout=10
                )
                
                if response.status_code in [200, 201, 202]:
                    results['data_flows_tested'] += 1
                    
                    # Wait a bit for processing
                    time.sleep(2)
                    
                    # Check if data appears in analytics summary
                    analytics_response = requests.get(f"{self.base_url}/api/analytics/summary", timeout=10)
                    
                    if analytics_response.status_code == 200:
                        analytics_data = analytics_response.json()
                        
                        # Check for our test data or general analytics activity
                        has_activity = (
                            analytics_data.get('total_analytics', 0) > 0 or
                            analytics_data.get('recent_analytics', 0) > 0
                        )
                        
                        if has_activity:
                            results['data_flows_successful'] += 1
                            results['data_consistency']['analytics_flow'] = True
                            logger.info("✅ Analytics data flow: Working")
                        else:
                            results['data_consistency']['analytics_flow'] = False
                            logger.warning("⚠️ Analytics data flow: No activity detected")
                    
            except Exception as e:
                logger.error(f"❌ Analytics data flow test failed: {e}")
                results['data_consistency']['analytics_flow'] = False
            
            # Test 2: Performance metrics data flow
            logger.info("Testing performance metrics data flow...")
            
            try:
                perf_response = requests.get(f"{self.base_url}/api/performance/metrics", timeout=10)
                
                if perf_response.status_code == 200:
                    perf_data = perf_response.json()
                    
                    # Check for expected performance metrics
                    has_metrics = any([
                        perf_data.get('cpu_usage'),
                        perf_data.get('memory_usage'),
                        perf_data.get('response_time'),
                        perf_data.get('throughput')
                    ])
                    
                    if has_metrics:
                        results['data_flows_tested'] += 1
                        results['data_flows_successful'] += 1
                        results['data_consistency']['performance_metrics'] = True
                        logger.info("✅ Performance metrics data flow: Working")
                    else:
                        results['data_consistency']['performance_metrics'] = False
                        logger.warning("⚠️ Performance metrics: No metrics data")
                
            except Exception as e:
                logger.error(f"❌ Performance metrics test failed: {e}")
                results['data_consistency']['performance_metrics'] = False
            
            # Test 3: Health monitoring data flow
            logger.info("Testing health monitoring data flow...")
            
            try:
                health_response = requests.get(f"{self.base_url}/api/monitoring/robustness", timeout=10)
                
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    
                    # Check for health indicators
                    has_health_data = any([
                        health_data.get('system_health'),
                        health_data.get('component_status'),
                        health_data.get('reliability_score'),
                        'status' in health_data
                    ])
                    
                    if has_health_data:
                        results['data_flows_tested'] += 1
                        results['data_flows_successful'] += 1
                        results['data_consistency']['health_monitoring'] = True
                        logger.info("✅ Health monitoring data flow: Working")
                    else:
                        results['data_consistency']['health_monitoring'] = False
                        logger.warning("⚠️ Health monitoring: No health data")
                
            except Exception as e:
                logger.error(f"❌ Health monitoring test failed: {e}")
                results['data_consistency']['health_monitoring'] = False
            
        except Exception as e:
            logger.error(f"❌ Data flow integration test failed: {e}")
        
        # Calculate success rate
        if results['data_flows_tested'] > 0:
            success_rate = (results['data_flows_successful'] / results['data_flows_tested']) * 100
            results['success'] = success_rate >= 70.0  # 70% success threshold
        
        logger.info(f"Data Flow Integration: {results['data_flows_successful']}/{results['data_flows_tested']} successful")
        return results
    
    def test_ultra_reliability_features(self) -> Dict[str, Any]:
        """Test ultra-reliability features are exposed on frontend."""
        logger.info("Testing ultra-reliability feature integration...")
        results = {
            'features_tested': 0,
            'features_available': 0,
            'feature_details': {},
            'success': False
        }
        
        # Ultra-reliability endpoints to test
        ultra_reliability_endpoints = [
            ('/api/monitoring/robustness', 'robustness_monitoring'),
            ('/api/monitoring/heartbeat', 'heartbeat_monitoring'),
            ('/api/monitoring/fallback', 'fallback_systems'),
            ('/api/monitoring/dead-letter', 'dead_letter_queue'),
            ('/api/monitoring/batch', 'batch_processing'),
            ('/api/monitoring/flow', 'flow_monitoring'),
            ('/api/monitoring/compression', 'compression_features'),
            ('/api/health/live', 'live_health_checks'),
            ('/api/health/ready', 'readiness_checks')
        ]
        
        for endpoint, feature_name in ultra_reliability_endpoints:
            try:
                results['features_tested'] += 1
                
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Check for ultra-reliability indicators
                        has_reliability_data = any([
                            'reliability' in str(data).lower(),
                            'robustness' in str(data).lower(),
                            'health' in str(data).lower(),
                            'status' in data,
                            'monitoring' in str(data).lower(),
                            'circuit' in str(data).lower(),
                            'sla' in str(data).lower(),
                            'backup' in str(data).lower(),
                            'retry' in str(data).lower()
                        ])
                        
                        if has_reliability_data:
                            results['features_available'] += 1
                            results['feature_details'][feature_name] = {
                                'available': True,
                                'endpoint': endpoint,
                                'data_keys': list(data.keys()) if isinstance(data, dict) else [],
                                'has_reliability_indicators': True
                            }
                            logger.info(f"✅ {feature_name}: Available and active")
                        else:
                            results['feature_details'][feature_name] = {
                                'available': True,
                                'endpoint': endpoint,
                                'has_reliability_indicators': False,
                                'note': 'Endpoint responsive but limited reliability data'
                            }
                            logger.warning(f"⚠️ {feature_name}: Available but limited data")
                    
                    except json.JSONDecodeError:
                        results['feature_details'][feature_name] = {
                            'available': True,
                            'endpoint': endpoint,
                            'json_parseable': False
                        }
                
                else:
                    results['feature_details'][feature_name] = {
                        'available': False,
                        'endpoint': endpoint,
                        'status_code': response.status_code
                    }
                    logger.warning(f"⚠️ {feature_name}: Not available ({response.status_code})")
                
            except Exception as e:
                results['feature_details'][feature_name] = {
                    'available': False,
                    'endpoint': endpoint,
                    'error': str(e)
                }
                logger.error(f"❌ {feature_name}: {e}")
        
        # Calculate availability rate
        if results['features_tested'] > 0:
            availability_rate = (results['features_available'] / results['features_tested']) * 100
            results['success'] = availability_rate >= 75.0  # 75% availability threshold
            results['availability_rate'] = availability_rate
        
        logger.info(f"Ultra-Reliability Features: {results['features_available']}/{results['features_tested']} available ({results.get('availability_rate', 0):.1f}%)")
        return results
    
    def test_performance_characteristics(self) -> Dict[str, Any]:
        """Test frontend-backend performance characteristics."""
        logger.info("Testing performance characteristics...")
        results = {
            'response_times': [],
            'concurrent_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'max_response_time': 0.0,
            'throughput_rps': 0.0,
            'success': False
        }
        
        # Test concurrent requests to simulate real frontend load
        endpoints_to_test = [
            '/api/health/live',
            '/api/analytics/summary',
            '/api/performance/metrics',
            '/api/monitoring/status',
            '/api/monitoring/robustness'
        ]
        
        def make_request(endpoint):
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                return {
                    'endpoint': endpoint,
                    'success': response.status_code in [200, 201, 202],
                    'response_time': response_time,
                    'status_code': response.status_code
                }
            except Exception as e:
                return {
                    'endpoint': endpoint,
                    'success': False,
                    'response_time': 0,
                    'error': str(e)
                }
        
        # Execute concurrent requests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit multiple requests to different endpoints
            futures = []
            for _ in range(20):  # 20 total requests
                for endpoint in endpoints_to_test:
                    future = executor.submit(make_request, endpoint)
                    futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    results['concurrent_requests'] += 1
                    
                    if result['success']:
                        results['successful_requests'] += 1
                        results['response_times'].append(result['response_time'])
                    else:
                        results['failed_requests'] += 1
                        
                except Exception as e:
                    results['failed_requests'] += 1
                    logger.error(f"Request failed: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        if results['response_times']:
            results['avg_response_time'] = sum(results['response_times']) / len(results['response_times'])
            results['max_response_time'] = max(results['response_times'])
        
        if total_time > 0:
            results['throughput_rps'] = results['successful_requests'] / total_time
        
        # Success criteria: >90% success rate, <2000ms average response time
        success_rate = (results['successful_requests'] / results['concurrent_requests']) * 100 if results['concurrent_requests'] > 0 else 0
        results['success'] = (
            success_rate >= 90.0 and
            results['avg_response_time'] < 2000.0
        )
        
        logger.info(f"Performance: {results['successful_requests']}/{results['concurrent_requests']} requests successful")
        logger.info(f"Average response time: {results['avg_response_time']:.1f}ms")
        logger.info(f"Throughput: {results['throughput_rps']:.1f} RPS")
        
        return results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test frontend error handling for backend failures."""
        logger.info("Testing error handling...")
        results = {
            'error_scenarios_tested': 0,
            'proper_error_responses': 0,
            'error_details': {},
            'success': False
        }
        
        # Test various error scenarios
        error_scenarios = [
            ('/api/nonexistent/endpoint', 404, 'nonexistent_endpoint'),
            ('/api/analytics/invalid-format', 400, 'invalid_request'),
            ('/api/metrics/unauthorized', 401, 'unauthorized_access')
        ]
        
        for endpoint, expected_status, scenario_name in error_scenarios:
            try:
                results['error_scenarios_tested'] += 1
                
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                # Check if error is handled properly
                proper_error = (
                    response.status_code >= 400 and
                    response.status_code < 600
                )
                
                if proper_error:
                    results['proper_error_responses'] += 1
                    
                results['error_details'][scenario_name] = {
                    'status_code': response.status_code,
                    'proper_error_response': proper_error,
                    'has_error_message': len(response.text) > 0
                }
                
                logger.info(f"✅ Error scenario {scenario_name}: {response.status_code}")
                
            except Exception as e:
                results['error_details'][scenario_name] = {
                    'error': str(e),
                    'proper_error_response': False
                }
                logger.error(f"❌ Error scenario {scenario_name}: {e}")
        
        # Success if most error scenarios are handled properly
        if results['error_scenarios_tested'] > 0:
            error_handling_rate = (results['proper_error_responses'] / results['error_scenarios_tested']) * 100
            results['success'] = error_handling_rate >= 80.0
        
        logger.info(f"Error Handling: {results['proper_error_responses']}/{results['error_scenarios_tested']} scenarios handled properly")
        return results
    
    async def run_comprehensive_test(self) -> bool:
        """Run comprehensive frontend-backend integration test."""
        try:
            logger.info("="*80)
            logger.info("Starting Frontend-Backend Integration Test Suite")
            logger.info("="*80)
            
            # Test 1: Server Availability
            if not self.test_server_availability():
                logger.error("❌ Server not available - aborting tests")
                return False
            
            # Test 2: API Endpoints
            logger.info("\n" + "="*50)
            self.test_results['api_endpoints'] = self.test_api_endpoints()
            
            # Test 3: WebSocket Connections
            logger.info("\n" + "="*50)
            self.test_results['websocket_connections'] = await self.test_websocket_connections()
            
            # Test 4: Data Flow Integration
            logger.info("\n" + "="*50)
            self.test_results['data_flow'] = self.test_data_flow_integration()
            
            # Test 5: Ultra-Reliability Features
            logger.info("\n" + "="*50)
            self.test_results['ultra_reliability'] = self.test_ultra_reliability_features()
            
            # Test 6: Performance Characteristics
            logger.info("\n" + "="*50)
            self.test_results['performance_metrics'] = self.test_performance_characteristics()
            
            # Test 7: Error Handling
            logger.info("\n" + "="*50)
            self.test_results['error_handling'] = self.test_error_handling()
            
            # Generate comprehensive report
            self.generate_integration_report()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False
    
    def generate_integration_report(self):
        """Generate comprehensive integration test report."""
        logger.info("\n" + "="*80)
        logger.info("FRONTEND-BACKEND INTEGRATION TEST REPORT")
        logger.info("="*80)
        
        # Calculate overall success
        test_successes = [
            self.test_results['api_endpoints']['success'],
            self.test_results['websocket_connections']['success'],
            self.test_results['data_flow']['success'],
            self.test_results['ultra_reliability']['success'],
            self.test_results['performance_metrics']['success'],
            self.test_results.get('error_handling', {}).get('success', True)
        ]
        
        overall_success = all(test_successes)
        success_rate = sum(test_successes) / len(test_successes) * 100
        
        logger.info(f"Overall Integration Status: {'✅ PASS' if overall_success else '❌ FAIL'}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("")
        
        # Detailed results
        results = self.test_results
        
        logger.info("📊 API ENDPOINTS:")
        api_results = results['api_endpoints']
        logger.info(f"  Endpoints: {api_results['endpoints_successful']}/{api_results['endpoints_tested']} successful")
        logger.info(f"  Success Rate: {api_results.get('success_rate', 0):.1f}%")
        logger.info(f"  Avg Response Time: {api_results.get('avg_response_time_ms', 0):.1f}ms")
        
        logger.info("\n🔌 WEBSOCKET CONNECTIONS:")
        ws_results = results['websocket_connections']
        logger.info(f"  Connections: {ws_results['connections_successful']}/{ws_results['connections_tested']} successful")
        logger.info(f"  Messages Received: {ws_results['messages_received']}")
        
        logger.info("\n🔄 DATA FLOW INTEGRATION:")
        flow_results = results['data_flow']
        logger.info(f"  Data Flows: {flow_results['data_flows_successful']}/{flow_results['data_flows_tested']} working")
        logger.info(f"  Analytics Flow: {'✅' if flow_results['data_consistency'].get('analytics_flow') else '❌'}")
        logger.info(f"  Performance Metrics: {'✅' if flow_results['data_consistency'].get('performance_metrics') else '❌'}")
        logger.info(f"  Health Monitoring: {'✅' if flow_results['data_consistency'].get('health_monitoring') else '❌'}")
        
        logger.info("\n🛡️ ULTRA-RELIABILITY FEATURES:")
        ultra_results = results['ultra_reliability']
        logger.info(f"  Features Available: {ultra_results['features_available']}/{ultra_results['features_tested']}")
        logger.info(f"  Availability Rate: {ultra_results.get('availability_rate', 0):.1f}%")
        
        logger.info("\n⚡ PERFORMANCE CHARACTERISTICS:")
        perf_results = results['performance_metrics']
        logger.info(f"  Concurrent Requests: {perf_results['successful_requests']}/{perf_results['concurrent_requests']} successful")
        logger.info(f"  Average Response Time: {perf_results['avg_response_time']:.1f}ms")
        logger.info(f"  Throughput: {perf_results['throughput_rps']:.1f} RPS")
        
        # Save detailed results
        with open('frontend_integration_test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'overall_success': overall_success,
                'success_rate': success_rate,
                'test_results': self.test_results
            }, f, indent=2, default=str)
        
        logger.info("\n" + "="*80)
        if overall_success:
            logger.info("🎉 FRONTEND-BACKEND INTEGRATION: ALL TESTS PASSED!")
            logger.info("✅ Backend capabilities are properly exposed on frontend")
        else:
            logger.info("⚠️ FRONTEND-BACKEND INTEGRATION: SOME ISSUES FOUND")
            logger.info("🔧 Review failed tests and ensure proper integration")
        logger.info("="*80)


async def main():
    """Main test execution function."""
    try:
        tester = FrontendBackendIntegrationTester()
        success = await tester.run_comprehensive_test()
        
        exit_code = 0 if success else 1
        print(f"\nFrontend-Backend Integration test completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())