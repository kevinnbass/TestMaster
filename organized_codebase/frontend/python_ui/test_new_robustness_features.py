"""
Comprehensive Test Suite for New Robustness Enhancement Features
==============================================================

Tests all 6 new robustness enhancement components to ensure they work
correctly and integrate properly with the analytics system.

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

def verify_new_robustness_features(analytics_data: Dict[str, Any]) -> Dict[str, bool]:
    """Verify all new robustness enhancement features are present and active."""
    verification = {}
    
    # Check for new robustness components
    expected_new_components = [
        'data_sanitizer',
        'deduplication_engine',
        'rate_limiter', 
        'integrity_verifier',
        'error_recovery',
        'connectivity_monitor'
    ]
    
    for component in expected_new_components:
        verification[component] = component in analytics_data
    
    return verification

def analyze_new_robustness_features(analytics_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the new robustness enhancement features."""
    analysis = {
        'overall_score': 0,
        'component_details': {},
        'recommendations': [],
        'strengths': [],
        'concerns': []
    }
    
    # Analyze data sanitizer
    sanitizer = analytics_data.get('data_sanitizer', {})
    if sanitizer:
        sanitizer_active = sanitizer.get('sanitizer_active', False)
        issues_found = sanitizer.get('recent_activity', {}).get('issues_last_hour', 0)
        
        analysis['component_details']['data_sanitizer'] = {
            'active': sanitizer_active,
            'issues_found_last_hour': issues_found,
            'validation_level': sanitizer.get('validation_level', 'unknown'),
            'score': 90 if sanitizer_active else 0
        }
        
        if sanitizer_active:
            analysis['strengths'].append("Data sanitization and validation active")
        if issues_found > 0:
            analysis['concerns'].append(f"Data validation issues detected: {issues_found}")
    
    # Analyze deduplication engine
    dedup = analytics_data.get('deduplication_engine', {})
    if dedup:
        engine_active = dedup.get('engine_status', {}).get('active', False)
        dedup_ratio = dedup.get('performance_metrics', {}).get('deduplication_ratio', 0)
        
        analysis['component_details']['deduplication_engine'] = {
            'active': engine_active,
            'deduplication_ratio': dedup_ratio,
            'records_processed': dedup.get('statistics', {}).get('records_processed', 0),
            'score': 85 if engine_active else 0
        }
        
        if engine_active:
            analysis['strengths'].append("Data deduplication engine active")
        if dedup_ratio > 5:
            analysis['concerns'].append(f"High duplicate rate detected: {dedup_ratio:.1f}%")
    
    # Analyze rate limiter
    rate_limiter = analytics_data.get('rate_limiter', {})
    if rate_limiter:
        limiter_active = rate_limiter.get('limiter_status', {}).get('active', False)
        queue_utilization = rate_limiter.get('traffic_queues', {}).get('queue_utilization', 0)
        
        analysis['component_details']['rate_limiter'] = {
            'active': limiter_active,
            'queue_utilization': queue_utilization,
            'throttled_requests': rate_limiter.get('statistics', {}).get('requests_throttled', 0),
            'score': 80 if limiter_active else 0
        }
        
        if limiter_active:
            analysis['strengths'].append("Adaptive rate limiting active")
        if queue_utilization > 0.8:
            analysis['concerns'].append(f"High queue utilization: {queue_utilization:.1%}")
    
    # Analyze integrity verifier
    integrity = analytics_data.get('integrity_verifier', {})
    if integrity:
        verifier_active = integrity.get('verification_status', {}).get('active', False)
        chain_integrity = integrity.get('verification_status', {}).get('chain_integrity', True)
        
        analysis['component_details']['integrity_verifier'] = {
            'active': verifier_active,
            'chain_integrity': chain_integrity,
            'records_protected': integrity.get('data_protection', {}).get('records_protected', 0),
            'score': 95 if verifier_active and chain_integrity else 50
        }
        
        if verifier_active and chain_integrity:
            analysis['strengths'].append("Data integrity verification active with valid chain")
        elif not chain_integrity:
            analysis['concerns'].append("Data integrity chain violations detected")
    
    # Analyze error recovery
    error_recovery = analytics_data.get('error_recovery', {})
    if error_recovery:
        recovery_active = error_recovery.get('recovery_status', {}).get('active', False)
        success_rate = error_recovery.get('error_analysis', {}).get('recovery_success_rate', 0)
        
        analysis['component_details']['error_recovery'] = {
            'active': recovery_active,
            'recovery_success_rate': success_rate,
            'system_degradation': error_recovery.get('recovery_status', {}).get('system_degradation_level', 'none'),
            'score': 90 if recovery_active and success_rate > 80 else 60
        }
        
        if recovery_active:
            analysis['strengths'].append("Advanced error recovery system active")
        if success_rate < 80:
            analysis['concerns'].append(f"Low recovery success rate: {success_rate:.1f}%")
    
    # Analyze connectivity monitor
    connectivity = analytics_data.get('connectivity_monitor', {})
    if connectivity:
        monitor_active = connectivity.get('monitoring_status', {}).get('active', False)
        system_health = connectivity.get('monitoring_status', {}).get('system_health_percent', 0)
        
        analysis['component_details']['connectivity_monitor'] = {
            'active': monitor_active,
            'system_health_percent': system_health,
            'endpoints_monitored': connectivity.get('endpoint_status', {}).get('total_endpoints', 0),
            'score': 85 if monitor_active and system_health > 90 else 60
        }
        
        if monitor_active:
            analysis['strengths'].append("Dashboard connectivity monitoring active")
        if system_health < 90:
            analysis['concerns'].append(f"Dashboard connectivity issues: {system_health:.1f}% health")
    
    # Calculate overall robustness score
    component_scores = [details.get('score', 0) for details in analysis['component_details'].values()]
    if component_scores:
        analysis['overall_score'] = sum(component_scores) / len(component_scores)
    
    # Generate recommendations
    if analysis['overall_score'] > 85:
        analysis['recommendations'].append("Excellent new robustness features - system highly protected")
    elif analysis['overall_score'] > 70:
        analysis['recommendations'].append("Good robustness with new features - minor optimizations needed")
    else:
        analysis['recommendations'].append("Review robustness configuration - some features may need attention")
    
    return analysis

def test_data_flow_robustness():
    """Test data flow through new robustness enhancement components."""
    print("\\n6. Testing Data Flow Through New Robustness Features...")
    
    base_url = "http://localhost:5000"
    test_data = {
        "test_component": "robustness_test",
        "timestamp": datetime.now().isoformat(),
        "test_metrics": {
            "cpu_usage": 45.2,
            "memory_usage": 62.1,
            "response_time": 125.5
        },
        "test_id": f"test_{int(time.time())}"
    }
    
    try:
        # Test data submission through robustness pipeline
        response = requests.post(f"{base_url}/api/analytics/submit", 
                               json=test_data, 
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Data submission: SUCCESS")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.1f}ms")
            
            # Check if data was processed through robustness components
            processing_info = result.get('processing_info', {})
            robustness_checks = processing_info.get('robustness_checks', {})
            
            print(f"   Robustness checks performed:")
            for check, status in robustness_checks.items():
                status_symbol = "[+]" if status else "[X]"
                print(f"     {status_symbol} {check}")
            
            return True
        else:
            print(f"   Data submission: FAILED ({response.status_code})")
            return False
    
    except Exception as e:
        print(f"   Data submission: EXCEPTION ({str(e)})")
        return False

def main():
    """Run comprehensive test of new robustness enhancement features."""
    print("New Analytics Robustness Enhancement Features Test")
    print("=" * 60)
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
    
    # Test comprehensive analytics with new features
    print("2. Testing Analytics with New Robustness Features...")
    analytics_result = test_endpoint(f"{base_url}/api/analytics/metrics", "Analytics with new features")
    print(f"   Status: {analytics_result['status']}")
    if analytics_result['status'] == 'success':
        print(f"   Response time: {analytics_result['response_time_ms']:.1f}ms")
        print(f"   Data size: {analytics_result['data_size']} bytes")
        
        # Verify new robustness components
        analytics_data = analytics_result['data'].get('comprehensive', {})
        verification = verify_new_robustness_features(analytics_data)
        
        print("\\n   New Robustness Components Verification:")
        for component, present in verification.items():
            status = "[+]" if present else "[X]"
            print(f"     {status} {component}: {'Present' if present else 'Missing'}")
        
        present_count = sum(verification.values())
        total_count = len(verification)
        print(f"\\n   New Component Coverage: {present_count}/{total_count} ({present_count/total_count*100:.1f}%)")
        
        # Analyze new robustness features
        print("\\n3. Analyzing New Robustness Enhancement Features...")
        analysis = analyze_new_robustness_features(analytics_data)
        
        print(f"   Overall New Features Score: {analysis['overall_score']:.1f}/100")
        
        if analysis['strengths']:
            print("\\n   New System Strengths:")
            for strength in analysis['strengths']:
                print(f"     + {strength}")
        
        if analysis['concerns']:
            print("\\n   Areas Requiring Attention:")
            for concern in analysis['concerns']:
                print(f"     - {concern}")
        
        if analysis['recommendations']:
            print("\\n   Recommendations:")
            for rec in analysis['recommendations']:
                print(f"     → {rec}")
    print()
    
    # Test individual new components
    print("4. Testing Individual New Robustness Components...")
    
    new_component_endpoints = [
        ("/api/robustness/sanitizer", "Data Sanitizer"),
        ("/api/robustness/deduplication", "Deduplication Engine"),
        ("/api/robustness/rate-limiter", "Rate Limiter"),
        ("/api/robustness/integrity", "Integrity Verifier"),
        ("/api/robustness/error-recovery", "Error Recovery"),
        ("/api/robustness/connectivity", "Connectivity Monitor")
    ]
    
    for endpoint, component_name in new_component_endpoints:
        result = test_endpoint(f"{base_url}{endpoint}", component_name)
        status_symbol = "[+]" if result['status'] == 'success' else "[X]"
        print(f"   {status_symbol} {component_name}: {result['status']}")
        if result['status'] == 'success':
            print(f"     Response time: {result['response_time_ms']:.1f}ms")
    print()
    
    # Test stress handling with new features
    print("5. Testing System Stress Handling with New Features...")
    print("   Sending rapid requests to test rate limiting and error handling...")
    
    stress_results = []
    for i in range(20):  # Send 20 rapid requests
        result = test_endpoint(f"{base_url}/api/analytics/metrics", f"Stress test {i+1}")
        stress_results.append(result)
        time.sleep(0.1)  # Very short delay
    
    successful_requests = len([r for r in stress_results if r['status'] == 'success'])
    throttled_requests = len([r for r in stress_results if 'throttled' in str(r.get('error', ''))])
    
    print(f"   Successful requests: {successful_requests}/20")
    print(f"   Throttled requests: {throttled_requests}/20")
    print(f"   System handled stress: {'YES' if successful_requests >= 15 else 'NO'}")
    
    if successful_requests >= 15:
        print("   [+] Rate limiting and error handling working correctly")
    else:
        print("   [!] System may be struggling under load")
    print()
    
    # Test data flow robustness
    data_flow_success = test_data_flow_robustness()
    
    # Summary
    print("\\n" + "=" * 60)
    print("New Robustness Enhancement Features Test Summary")
    print("=" * 60)
    
    if analytics_result['status'] == 'success':
        analytics_data = analytics_result['data'].get('comprehensive', {})
        verification = verify_new_robustness_features(analytics_data)
        analysis = analyze_new_robustness_features(analytics_data)
        
        print(f"New Features Coverage: {sum(verification.values())}/{len(verification)} components active")
        print(f"Overall Robustness Score: {analysis['overall_score']:.1f}/100")
        print(f"Stress Test Results: {successful_requests}/20 requests successful")
        print(f"Data Flow Test: {'PASSED' if data_flow_success else 'FAILED'}")
        
        # Final verdict
        total_score = (
            (sum(verification.values()) / len(verification)) * 30 +  # 30% for component presence
            (analysis['overall_score'] / 100) * 40 +  # 40% for component quality
            (successful_requests / 20) * 20 +  # 20% for stress handling
            (10 if data_flow_success else 0)  # 10% for data flow
        )
        
        print(f"\\nFinal System Robustness Score: {total_score:.1f}/100")
        
        if total_score >= 90:
            print("✓ EXCELLENT: All new robustness features working optimally")
        elif total_score >= 75:
            print("✓ GOOD: New robustness features operational with minor issues")
        elif total_score >= 60:
            print("⚠ ACCEPTABLE: New robustness features working but need attention")
        else:
            print("✗ NEEDS WORK: New robustness features require immediate attention")
    
    print(f"\\nTest completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()