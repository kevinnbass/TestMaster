"""
Analytics Robustness System Verification Test
============================================

Comprehensive test suite to verify all robustness enhancement features
are working correctly and providing improved analytics reliability.

Author: TestMaster Team
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Any

def test_endpoint(url: str, description: str) -> Dict[str, Any]:
    """Test an endpoint and return results."""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=30)
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            try:
                data = response.json()
                return {
                    'status': 'success',
                    'description': description,
                    'response_time_ms': response_time,
                    'data_size': len(str(data)),
                    'data': data
                }
            except json.JSONDecodeError:
                return {
                    'status': 'json_error',
                    'description': description,
                    'response_time_ms': response_time,
                    'error': 'Invalid JSON response'
                }
        else:
            return {
                'status': 'http_error',
                'description': description,
                'response_time_ms': response_time,
                'status_code': response.status_code,
                'error': response.text[:200]
            }
    except Exception as e:
        return {
            'status': 'exception',
            'description': description,
            'error': str(e)
        }

def verify_robustness_components(analytics_data: Dict[str, Any]) -> Dict[str, bool]:
    """Verify all robustness components are present and active."""
    verification = {}
    
    # Check for all expected robustness components
    expected_components = [
        'redundancy_status',
        'watchdog_status', 
        'telemetry_status',
        'performance_optimizer',
        'quality_assurance',
        'circuit_breaker_status',
        'metrics_collection',
        'health_status',
        'streaming_stats',
        'persistence_stats',
        'cache_performance',
        'pipeline_stats',
        'normalization_stats'
    ]
    
    for component in expected_components:
        verification[component] = component in analytics_data
    
    return verification

def analyze_system_robustness(analytics_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the robustness characteristics of the system."""
    analysis = {
        'overall_score': 0,
        'component_details': {},
        'recommendations': [],
        'strengths': [],
        'concerns': []
    }
    
    # Analyze redundancy
    redundancy = analytics_data.get('redundancy_status', {})
    if redundancy:
        active_nodes = redundancy.get('active_nodes', 0)
        total_nodes = redundancy.get('total_nodes', 0)
        failover_mode = redundancy.get('failover_mode', 'unknown')
        
        analysis['component_details']['redundancy'] = {
            'active_nodes': active_nodes,
            'total_nodes': total_nodes,
            'failover_mode': failover_mode,
            'score': min(100, (active_nodes / max(1, total_nodes)) * 100)
        }
        
        if active_nodes >= 2:
            analysis['strengths'].append(f"Multiple redundancy nodes active ({active_nodes})")
        else:
            analysis['concerns'].append("Limited redundancy - only one active node")
    
    # Analyze watchdog
    watchdog = analytics_data.get('watchdog_status', {})
    if watchdog:
        monitored_components = watchdog.get('total_components', 0)
        healthy_components = watchdog.get('healthy_components', 0)
        critical_components = watchdog.get('critical_components', 0)
        
        analysis['component_details']['watchdog'] = {
            'monitored_components': monitored_components,
            'healthy_components': healthy_components,
            'critical_components': critical_components,
            'score': min(100, (healthy_components / max(1, monitored_components)) * 100)
        }
        
        if healthy_components == monitored_components:
            analysis['strengths'].append("All monitored components are healthy")
        else:
            analysis['concerns'].append(f"Some components unhealthy ({monitored_components - healthy_components})")
    
    # Analyze telemetry
    telemetry = analytics_data.get('telemetry_status', {})
    if telemetry:
        collection_active = telemetry.get('collection_status', {}).get('active', False)
        events_collected = telemetry.get('statistics', {}).get('events_collected', 0)
        spans_created = telemetry.get('statistics', {}).get('spans_created', 0)
        
        analysis['component_details']['telemetry'] = {
            'collection_active': collection_active,
            'events_collected': events_collected,
            'spans_created': spans_created,
            'score': 100 if collection_active else 0
        }
        
        if collection_active:
            analysis['strengths'].append("Comprehensive telemetry collection active")
    
    # Analyze performance optimizer
    optimizer = analytics_data.get('performance_optimizer', {})
    if optimizer:
        optimizer_active = optimizer.get('optimizer_active', False)
        optimizations_applied = optimizer.get('statistics', {}).get('optimizations_applied', 0)
        performance_improvements = optimizer.get('statistics', {}).get('performance_improvements', 0)
        
        analysis['component_details']['performance_optimizer'] = {
            'optimizer_active': optimizer_active,
            'optimizations_applied': optimizations_applied,
            'performance_improvements': performance_improvements,
            'score': 100 if optimizer_active else 0
        }
        
        if optimizer_active:
            analysis['strengths'].append("Automated performance optimization active")
    
    # Analyze cache performance
    cache_perf = analytics_data.get('cache_performance', {})
    if cache_perf:
        hit_rate = cache_perf.get('efficiency', {}).get('hit_rate', 0)
        cache_score = min(100, hit_rate * 100)
        
        analysis['component_details']['cache'] = {
            'hit_rate': hit_rate,
            'score': cache_score
        }
        
        if hit_rate > 0.8:
            analysis['strengths'].append(f"High cache hit rate ({hit_rate:.1%})")
        elif hit_rate < 0.5:
            analysis['concerns'].append(f"Low cache hit rate ({hit_rate:.1%})")
    
    # Calculate overall robustness score
    component_scores = [details.get('score', 0) for details in analysis['component_details'].values()]
    if component_scores:
        analysis['overall_score'] = sum(component_scores) / len(component_scores)
    
    # Generate recommendations based on analysis
    if analysis['overall_score'] > 90:
        analysis['recommendations'].append("System shows excellent robustness - maintain current configuration")
    elif analysis['overall_score'] > 70:
        analysis['recommendations'].append("Good robustness with room for optimization")
    else:
        analysis['recommendations'].append("Consider reviewing robustness configuration for improvements")
    
    return analysis

def main():
    """Run comprehensive robustness verification test."""
    print("Analytics Robustness System Verification")
    print("=" * 50)
    print(f"Test started at: {datetime.now().isoformat()}")
    print()
    
    base_url = "http://localhost:5000"
    
    # Test basic health
    print("1. Testing Basic System Health...")
    health_result = test_endpoint(f"{base_url}/api/health/live", "Basic health check")
    print(f"   Status: {health_result['status']}")
    if health_result['status'] == 'success':
        print(f"   Response time: {health_result['response_time_ms']:.1f}ms")
    print()
    
    # Test comprehensive analytics
    print("2. Testing Comprehensive Analytics with Robustness Features...")
    analytics_result = test_endpoint(f"{base_url}/api/analytics/metrics", "Comprehensive analytics")
    print(f"   Status: {analytics_result['status']}")
    if analytics_result['status'] == 'success':
        print(f"   Response time: {analytics_result['response_time_ms']:.1f}ms")
        print(f"   Data size: {analytics_result['data_size']} bytes")
        
        # Verify robustness components
        analytics_data = analytics_result['data'].get('comprehensive', {})
        verification = verify_robustness_components(analytics_data)
        
        print("\\n   Robustness Components Verification:")
        for component, present in verification.items():
            status = "[+]" if present else "[X]"
            print(f"     {status} {component}: {'Present' if present else 'Missing'}")
        
        present_count = sum(verification.values())
        total_count = len(verification)
        print(f"\\n   Component Coverage: {present_count}/{total_count} ({present_count/total_count*100:.1f}%)")
        
        # Analyze system robustness
        print("\\n3. Analyzing System Robustness...")
        analysis = analyze_system_robustness(analytics_data)
        
        print(f"   Overall Robustness Score: {analysis['overall_score']:.1f}/100")
        
        if analysis['strengths']:
            print("\\n   System Strengths:")
            for strength in analysis['strengths']:
                print(f"     + {strength}")
        
        if analysis['concerns']:
            print("\\n   Areas of Concern:")
            for concern in analysis['concerns']:
                print(f"     - {concern}")
        
        if analysis['recommendations']:
            print("\\n   Recommendations:")
            for rec in analysis['recommendations']:
                print(f"     → {rec}")
    print()
    
    # Test component health
    print("4. Testing Component Health Monitoring...")
    components_result = test_endpoint(f"{base_url}/api/health/components", "Component health monitoring")
    print(f"   Status: {components_result['status']}")
    if components_result['status'] == 'success':
        components = components_result['data'].get('components', {})
        healthy_count = sum(1 for comp in components.values() if comp.get('status') == 'healthy')
        total_count = len(components)
        print(f"   Component Health: {healthy_count}/{total_count} components healthy")
        
        for comp_name, comp_data in components.items():
            status = comp_data.get('status', 'unknown')
            print(f"     - {comp_name}: {status}")
    print()
    
    # Test Prometheus metrics
    print("5. Testing Prometheus Metrics Export...")
    try:
        response = requests.get(f"{base_url}/api/health/prometheus", timeout=10)
        if response.status_code == 200:
            metrics_text = response.text
            metric_lines = [line for line in metrics_text.split('\\n') if line and not line.startswith('#')]
            print(f"   Status: success")
            print(f"   Metrics exported: {len(metric_lines)} metrics")
            print("   Sample metrics:")
            for line in metric_lines[:5]:
                print(f"     {line}")
        else:
            print(f"   Status: http_error ({response.status_code})")
    except Exception as e:
        print(f"   Status: exception ({str(e)})")
    print()
    
    # Performance stress test
    print("6. Performance Stress Test...")
    print("   Testing system under repeated load...")
    response_times = []
    
    for i in range(10):
        result = test_endpoint(f"{base_url}/api/analytics/metrics", f"Load test {i+1}")
        if result['status'] == 'success':
            response_times.append(result['response_time_ms'])
        time.sleep(0.5)
    
    if response_times:
        avg_response = sum(response_times) / len(response_times)
        max_response = max(response_times)
        min_response = min(response_times)
        
        print(f"   Completed {len(response_times)}/10 requests successfully")
        print(f"   Average response time: {avg_response:.1f}ms")
        print(f"   Response time range: {min_response:.1f}ms - {max_response:.1f}ms")
        
        if avg_response < 1000:
            print("   [+] Performance: Excellent (< 1s)")
        elif avg_response < 5000:
            print("   [+] Performance: Good (< 5s)")
        else:
            print("   [!] Performance: Needs attention (> 5s)")
    print()
    
    print("Robustness Verification Complete!")
    print(f"Test completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()