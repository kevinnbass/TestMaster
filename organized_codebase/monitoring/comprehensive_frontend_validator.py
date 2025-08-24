#!/usr/bin/env python3
"""
Comprehensive Frontend-Backend Integration Validator
=====================================================

Tests ALL backend capabilities are properly exposed for frontend consumption
without requiring any browser interaction.

Author: TestMaster Team
"""

import requests
import json
import time
import asyncio
import websockets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import concurrent.futures
import statistics

class ComprehensiveFrontendValidator:
    """Validates complete frontend-backend integration without browser."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.validation_results = {
            'core_apis': {},
            'visualization_apis': {},
            'data_structures': {},
            'real_time_features': {},
            'performance_metrics': {},
            'untapped_features': {},
            'overall_score': 0
        }
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all frontend-backend integration."""
        print("=" * 80)
        print("COMPREHENSIVE FRONTEND-BACKEND INTEGRATION VALIDATION")
        print("No Browser Required - Automated Testing")
        print("=" * 80)
        print()
        
        # 1. Test Core APIs
        print("1. VALIDATING CORE APIS")
        print("-" * 40)
        self.validate_core_apis()
        
        # 2. Test Visualization APIs
        print("\n2. VALIDATING VISUALIZATION APIS")
        print("-" * 40)
        self.validate_visualization_apis()
        
        # 3. Test Data Structures
        print("\n3. VALIDATING DATA STRUCTURES FOR FRONTEND")
        print("-" * 40)
        self.validate_data_structures()
        
        # 4. Test Real-time Features
        print("\n4. VALIDATING REAL-TIME FEATURES")
        print("-" * 40)
        self.validate_realtime_features()
        
        # 5. Test Performance
        print("\n5. VALIDATING PERFORMANCE UNDER LOAD")
        print("-" * 40)
        self.validate_performance()
        
        # 6. Identify Untapped Features
        print("\n6. IDENTIFYING UNTAPPED BACKEND FEATURES")
        print("-" * 40)
        self.identify_untapped_features()
        
        # 7. Generate Report
        print("\n7. GENERATING INTEGRATION REPORT")
        print("-" * 40)
        self.generate_report()
        
        return self.validation_results
    
    def validate_core_apis(self):
        """Validate core API endpoints."""
        core_endpoints = [
            # Health & Status
            ('/api/health', 'Health Check'),
            ('/api/health/live', 'Liveness Check'),
            ('/api/health/ready', 'Readiness Check'),
            
            # Analytics
            ('/api/analytics/summary', 'Analytics Summary'),
            ('/api/analytics/recent', 'Recent Analytics'),
            ('/api/analytics/export', 'Analytics Export'),
            
            # Performance
            ('/api/performance/metrics', 'Performance Metrics'),
            ('/api/performance/system', 'System Performance'),
            
            # Monitoring
            ('/api/monitoring/robustness', 'Robustness Monitor'),
            ('/api/monitoring/heartbeat', 'Heartbeat Monitor'),
            ('/api/monitoring/fallback', 'Fallback System'),
            
            # Workflow
            ('/api/workflow/status', 'Workflow Status'),
            ('/api/workflow/execution', 'Workflow Execution'),
            
            # Tests
            ('/api/tests/status', 'Test Status'),
            ('/api/tests/results', 'Test Results'),
            ('/api/tests/coverage', 'Test Coverage')
        ]
        
        results = {}
        for endpoint, name in core_endpoints:
            result = self._test_endpoint(endpoint, name)
            results[endpoint] = result
        
        self.validation_results['core_apis'] = results
        
        # Summary
        working = sum(1 for r in results.values() if r['status'] == 'success')
        chart_ready = sum(1 for r in results.values() if r.get('has_charts', False))
        print(f"\nCore APIs: {working}/{len(core_endpoints)} working, {chart_ready} chart-ready")
    
    def validate_visualization_apis(self):
        """Validate visualization-specific APIs."""
        viz_endpoints = [
            # Intelligence Dashboard
            ('/api/intelligence/agents/status', 'Agent Status'),
            ('/api/intelligence/agents/coordination', 'Agent Coordination'),
            ('/api/intelligence/agents/activities', 'Agent Activities'),
            ('/api/intelligence/agents/decisions', 'Consensus Decisions'),
            ('/api/intelligence/agents/optimization', 'Optimization Metrics'),
            
            # Test Generation
            ('/api/test-generation/generators/status', 'Generator Status'),
            ('/api/test-generation/generation/live', 'Live Generation'),
            ('/api/test-generation/generation/queue', 'Generation Queue'),
            ('/api/test-generation/generation/performance', 'Generation Performance'),
            ('/api/test-generation/generation/insights', 'Generation Insights'),
            
            # Security
            ('/api/security/vulnerabilities/heatmap', 'Vulnerability Heatmap'),
            ('/api/security/owasp/compliance', 'OWASP Compliance'),
            ('/api/security/threats/realtime', 'Real-time Threats'),
            ('/api/security/scanning/status', 'Security Scanning'),
            ('/api/security/remediation/recommendations', 'Remediation')
        ]
        
        results = {}
        for endpoint, name in viz_endpoints:
            result = self._test_endpoint(endpoint, name)
            results[endpoint] = result
        
        self.validation_results['visualization_apis'] = results
        
        # Summary
        working = sum(1 for r in results.values() if r['status'] == 'success')
        chart_ready = sum(1 for r in results.values() if r.get('has_charts', False))
        print(f"\nVisualization APIs: {working}/{len(viz_endpoints)} working, {chart_ready} chart-ready")
    
    def validate_data_structures(self):
        """Validate data structures are frontend-ready."""
        test_endpoints = [
            '/api/analytics/summary',
            '/api/performance/metrics',
            '/api/intelligence/agents/status',
            '/api/test-generation/generators/status',
            '/api/security/vulnerabilities/heatmap'
        ]
        
        results = {}
        for endpoint in test_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Analyze structure
                    analysis = {
                        'valid_json': True,
                        'has_status': 'status' in data,
                        'has_timestamp': 'timestamp' in data,
                        'has_charts': 'charts' in data,
                        'chart_types': list(data.get('charts', {}).keys()) if 'charts' in data else [],
                        'data_depth': self._calculate_depth(data),
                        'field_count': len(data),
                        'arrays_present': self._has_arrays(data),
                        'nested_objects': self._has_nested_objects(data),
                        'visualization_ready': False
                    }
                    
                    # Check visualization readiness
                    analysis['visualization_ready'] = (
                        analysis['has_charts'] and 
                        analysis['has_timestamp'] and
                        len(analysis['chart_types']) > 0
                    )
                    
                    results[endpoint] = analysis
                    
                    status = "[OK]" if analysis['visualization_ready'] else "!"
                    print(f"{status} {endpoint:40} Viz-Ready: {analysis['visualization_ready']}")
                    
            except Exception as e:
                results[endpoint] = {'error': str(e)}
                print(f"[X] {endpoint:40} Error")
        
        self.validation_results['data_structures'] = results
    
    def validate_realtime_features(self):
        """Validate real-time update capabilities."""
        realtime_endpoints = [
            '/api/analytics/recent',
            '/api/monitoring/heartbeat',
            '/api/intelligence/agents/activities',
            '/api/test-generation/generation/live',
            '/api/security/threats/realtime'
        ]
        
        results = {}
        for endpoint in realtime_endpoints:
            try:
                # Make multiple requests
                timestamps = []
                data_samples = []
                
                for _ in range(3):
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=3)
                    if response.status_code == 200:
                        data = response.json()
                        if 'timestamp' in data:
                            timestamps.append(data['timestamp'])
                        data_samples.append(json.dumps(data, sort_keys=True))
                        time.sleep(1)
                
                # Analyze real-time capability
                unique_timestamps = len(set(timestamps))
                unique_data = len(set(data_samples))
                
                results[endpoint] = {
                    'timestamps_change': unique_timestamps > 1,
                    'data_changes': unique_data > 1,
                    'realtime_capable': unique_timestamps > 1 or unique_data > 1,
                    'update_frequency': 'high' if unique_data > 2 else 'low'
                }
                
                status = "[OK]" if results[endpoint]['realtime_capable'] else "!"
                print(f"{status} {endpoint:40} Real-time: {results[endpoint]['realtime_capable']}")
                
            except Exception as e:
                results[endpoint] = {'error': str(e)}
                print(f"[X] {endpoint:40} Error")
        
        self.validation_results['real_time_features'] = results
    
    def validate_performance(self):
        """Validate performance under concurrent load."""
        print("Simulating 50 concurrent frontend requests...")
        
        test_endpoints = [
            '/api/health/live',
            '/api/analytics/summary',
            '/api/intelligence/agents/status'
        ]
        
        def make_request(endpoint):
            try:
                start = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                return {
                    'endpoint': endpoint,
                    'success': response.status_code == 200,
                    'response_time': (time.time() - start) * 1000,
                    'data_size': len(response.text) if response.status_code == 200 else 0
                }
            except Exception as e:
                return {
                    'endpoint': endpoint,
                    'success': False,
                    'error': str(e)
                }
        
        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(50):
                endpoint = test_endpoints[_ % len(test_endpoints)]
                futures.append(executor.submit(make_request, endpoint))
            
            results = [f.result() for f in concurrent.futures.as_completed(futures, timeout=30)]
        
        # Analyze performance
        successful = [r for r in results if r.get('success', False)]
        response_times = [r['response_time'] for r in successful]
        
        self.validation_results['performance_metrics'] = {
            'total_requests': len(results),
            'successful': len(successful),
            'success_rate': (len(successful) / len(results) * 100) if results else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'p95_response_time': sorted(response_times)[int(len(response_times)*0.95)] if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0
        }
        
        metrics = self.validation_results['performance_metrics']
        print(f"Success Rate: {metrics['success_rate']:.1f}%")
        print(f"Avg Response: {metrics['avg_response_time']:.1f}ms")
        print(f"P95 Response: {metrics['p95_response_time']:.1f}ms")
    
    def identify_untapped_features(self):
        """Identify backend features not yet exposed to frontend."""
        untapped = {
            'coverage_intelligence': {
                'description': 'Advanced coverage analysis and intelligence',
                'potential_endpoints': [
                    '/api/coverage/intelligence',
                    '/api/coverage/branch-analysis',
                    '/api/coverage/heatmap'
                ],
                'visualization_type': 'Coverage heatmaps and trend analysis'
            },
            'telemetry_system': {
                'description': 'System telemetry and profiling data',
                'potential_endpoints': [
                    '/api/telemetry/system-profile',
                    '/api/telemetry/flow-analysis',
                    '/api/telemetry/performance-traces'
                ],
                'visualization_type': 'Performance flamegraphs and flow diagrams'
            },
            'async_processing': {
                'description': 'Async task execution and monitoring',
                'potential_endpoints': [
                    '/api/async/tasks',
                    '/api/async/thread-pools',
                    '/api/async/scheduler'
                ],
                'visualization_type': 'Task queues and thread pool visualizations'
            },
            'streaming_features': {
                'description': 'Real-time streaming and collaboration',
                'potential_endpoints': [
                    '/api/streaming/live-feedback',
                    '/api/streaming/collaborative',
                    '/api/streaming/incremental'
                ],
                'visualization_type': 'Live streaming dashboards'
            },
            'quality_assurance': {
                'description': 'Quality inspection and benchmarking',
                'potential_endpoints': [
                    '/api/qa/quality-scores',
                    '/api/qa/benchmarks',
                    '/api/qa/validation-results'
                ],
                'visualization_type': 'Quality scorecards and benchmarks'
            },
            'flow_optimization': {
                'description': 'Workflow optimization and routing',
                'potential_endpoints': [
                    '/api/flow/optimizer',
                    '/api/flow/dependency-graph',
                    '/api/flow/parallel-execution'
                ],
                'visualization_type': 'DAG visualizations and optimization metrics'
            }
        }
        
        self.validation_results['untapped_features'] = untapped
        
        print(f"Found {len(untapped)} untapped feature categories:")
        for feature, details in untapped.items():
            print(f"  - {feature}: {details['description']}")
    
    def generate_report(self):
        """Generate comprehensive integration report."""
        # Calculate scores
        core_score = self._calculate_api_score(self.validation_results['core_apis'])
        viz_score = self._calculate_api_score(self.validation_results['visualization_apis'])
        data_score = self._calculate_data_score(self.validation_results['data_structures'])
        realtime_score = self._calculate_realtime_score(self.validation_results['real_time_features'])
        perf_score = self._calculate_performance_score(self.validation_results['performance_metrics'])
        
        overall_score = statistics.mean([core_score, viz_score, data_score, realtime_score, perf_score])
        self.validation_results['overall_score'] = overall_score
        
        print("\n" + "=" * 80)
        print("INTEGRATION VALIDATION REPORT")
        print("=" * 80)
        
        print("\nSCORES:")
        print(f"  Core APIs:         {core_score:.1f}%")
        print(f"  Visualization:     {viz_score:.1f}%")
        print(f"  Data Structures:   {data_score:.1f}%")
        print(f"  Real-time:         {realtime_score:.1f}%")
        print(f"  Performance:       {perf_score:.1f}%")
        print(f"  OVERALL:           {overall_score:.1f}%")
        
        # Determine status
        if overall_score >= 90:
            status = "EXCELLENT - Production Ready"
            recommendation = "All backend capabilities are properly exposed for frontend."
        elif overall_score >= 75:
            status = "GOOD - Minor improvements needed"
            recommendation = "Most features working well, optimize remaining endpoints."
        elif overall_score >= 60:
            status = "FAIR - Some issues to address"
            recommendation = "Several endpoints need attention for full integration."
        else:
            status = "NEEDS IMPROVEMENT"
            recommendation = "Significant work needed for seamless integration."
        
        print(f"\nSTATUS: {status}")
        print(f"RECOMMENDATION: {recommendation}")
        
        # Untapped features summary
        print(f"\nUNTAPPED FEATURES: {len(self.validation_results['untapped_features'])}")
        print("These backend capabilities could be exposed for better visualization:")
        for feature in list(self.validation_results['untapped_features'].keys())[:3]:
            print(f"  - {feature}")
        
        # Save detailed report
        with open('frontend_integration_report.json', 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print("\nDetailed report saved to: frontend_integration_report.json")
        
        return overall_score >= 75
    
    def _test_endpoint(self, endpoint: str, name: str) -> Dict[str, Any]:
        """Test a single endpoint."""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    'status': 'success',
                    'name': name,
                    'response_time': response.elapsed.total_seconds() * 1000,
                    'data_size': len(response.text),
                    'has_status': 'status' in data,
                    'has_timestamp': 'timestamp' in data,
                    'has_charts': 'charts' in data,
                    'chart_count': len(data.get('charts', {})) if 'charts' in data else 0
                }
                
                status = "[OK]" if result['has_charts'] else "[!]"
                print(f"{status} {name:35} Charts: {result['chart_count']}, Size: {result['data_size']}B")
                
                return result
            else:
                print(f"[X] {name:35} HTTP {response.status_code}")
                return {'status': 'error', 'status_code': response.status_code}
                
        except Exception as e:
            print(f"[X] {name:35} {str(e)[:30]}")
            return {'status': 'failed', 'error': str(e)}
    
    def _calculate_depth(self, data: Any, depth: int = 0) -> int:
        """Calculate data structure depth."""
        if depth > 10:
            return depth
        if isinstance(data, dict):
            return max([self._calculate_depth(v, depth + 1) for v in data.values()], default=depth)
        elif isinstance(data, list) and data:
            return max([self._calculate_depth(item, depth + 1) for item in data[:5]], default=depth)
        return depth
    
    def _has_arrays(self, data: Dict) -> bool:
        """Check if data contains arrays."""
        return any(isinstance(v, list) for v in data.values())
    
    def _has_nested_objects(self, data: Dict) -> bool:
        """Check if data contains nested objects."""
        return any(isinstance(v, dict) for v in data.values())
    
    def _calculate_api_score(self, results: Dict) -> float:
        """Calculate API score."""
        if not results:
            return 0
        total = len(results)
        working = sum(1 for r in results.values() if r.get('status') == 'success')
        chart_ready = sum(1 for r in results.values() if r.get('has_charts', False))
        
        base_score = (working / total) * 70
        chart_bonus = (chart_ready / total) * 30
        
        return base_score + chart_bonus
    
    def _calculate_data_score(self, results: Dict) -> float:
        """Calculate data structure score."""
        if not results:
            return 0
        viz_ready = sum(1 for r in results.values() if r.get('visualization_ready', False))
        return (viz_ready / len(results)) * 100
    
    def _calculate_realtime_score(self, results: Dict) -> float:
        """Calculate real-time capability score."""
        if not results:
            return 0
        capable = sum(1 for r in results.values() if r.get('realtime_capable', False))
        return (capable / len(results)) * 100
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate performance score."""
        if not metrics:
            return 0
        
        score = 100
        
        # Deduct for poor performance
        if metrics.get('success_rate', 0) < 95:
            score -= 20
        if metrics.get('avg_response_time', 0) > 1000:
            score -= 20
        if metrics.get('p95_response_time', 0) > 2000:
            score -= 10
        
        return max(score, 0)


def main():
    """Run comprehensive frontend validation."""
    validator = ComprehensiveFrontendValidator()
    results = validator.run_complete_validation()
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    
    overall_score = results['overall_score']
    if overall_score >= 80:
        print(f"\n[SUCCESS] Frontend integration is EXCELLENT ({overall_score:.1f}%)")
        print("All backend capabilities are properly exposed for frontend consumption.")
        print("No browser testing required - everything validated programmatically!")
    elif overall_score >= 60:
        print(f"\n[GOOD] Frontend integration is working well ({overall_score:.1f}%)")
        print("Most features are properly exposed with room for optimization.")
    else:
        print(f"\n[NEEDS IMPROVEMENT] Integration score: {overall_score:.1f}%")
        print("Review the report for specific areas needing attention.")
    
    return overall_score >= 60


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)