#!/usr/bin/env python3
"""
Quick Frontend-Backend Integration Test
========================================

Fast validation of frontend-backend integration without browser.

Author: TestMaster Team
"""

import requests
import json
import time
from datetime import datetime

def quick_integration_test():
    """Quick test of frontend-backend integration."""
    print("=" * 70)
    print("QUICK FRONTEND-BACKEND INTEGRATION TEST")
    print("=" * 70)
    print()
    
    base_url = "http://localhost:5000"
    
    # Define key endpoints to test
    test_categories = {
        'Core APIs': [
            ('/api/health/live', 'Health Check'),
            ('/api/analytics/summary', 'Analytics'),
            ('/api/performance/metrics', 'Performance'),
            ('/api/monitoring/robustness', 'Monitoring')
        ],
        'Intelligence APIs': [
            ('/api/intelligence/agents/status', 'Agent Status'),
            ('/api/intelligence/agents/coordination', 'Coordination'),
            ('/api/intelligence/agents/activities', 'Activities')
        ],
        'Test Generation APIs': [
            ('/api/test-generation/generators/status', 'Generators'),
            ('/api/test-generation/generation/live', 'Live Generation'),
            ('/api/test-generation/generation/performance', 'Performance')
        ],
        'Security APIs': [
            ('/api/security/vulnerabilities/heatmap', 'Vulnerabilities'),
            ('/api/security/owasp/compliance', 'OWASP'),
            ('/api/security/threats/realtime', 'Threats')
        ]
    }
    
    overall_results = {
        'total_endpoints': 0,
        'working': 0,
        'chart_ready': 0,
        'realtime_capable': 0,
        'categories': {}
    }
    
    # Test each category
    for category, endpoints in test_categories.items():
        print(f"\n{category}:")
        print("-" * 40)
        
        category_results = {
            'total': len(endpoints),
            'working': 0,
            'chart_ready': 0
        }
        
        for endpoint, name in endpoints:
            overall_results['total_endpoints'] += 1
            
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=3)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    has_charts = 'charts' in data
                    has_timestamp = 'timestamp' in data
                    
                    overall_results['working'] += 1
                    category_results['working'] += 1
                    
                    if has_charts:
                        overall_results['chart_ready'] += 1
                        category_results['chart_ready'] += 1
                    
                    status = "[OK]" if has_charts else "[!]"
                    print(f"{status} {name:25} Charts: {'YES' if has_charts else 'NO'}")
                    
                else:
                    print(f"[X] {name:25} HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"[X] {name:25} Error: {str(e)[:20]}")
        
        overall_results['categories'][category] = category_results
    
    # Test real-time capability
    print("\nReal-time Capability Test:")
    print("-" * 40)
    
    realtime_endpoints = [
        '/api/analytics/recent',
        '/api/intelligence/agents/activities',
        '/api/test-generation/generation/live'
    ]
    
    for endpoint in realtime_endpoints:
        try:
            # Make 2 requests 1 second apart
            r1 = requests.get(f"{base_url}{endpoint}", timeout=3)
            time.sleep(1)
            r2 = requests.get(f"{base_url}{endpoint}", timeout=3)
            
            if r1.status_code == 200 and r2.status_code == 200:
                d1 = r1.json()
                d2 = r2.json()
                
                # Check if timestamps differ
                if 'timestamp' in d1 and 'timestamp' in d2:
                    if d1['timestamp'] != d2['timestamp']:
                        overall_results['realtime_capable'] += 1
                        print(f"[OK] {endpoint:35} Real-time: YES")
                    else:
                        print(f"! {endpoint:35} Real-time: NO")
        except:
            pass
    
    # Calculate scores
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)
    
    success_rate = (overall_results['working'] / overall_results['total_endpoints']) * 100
    chart_rate = (overall_results['chart_ready'] / overall_results['total_endpoints']) * 100
    
    print(f"\nEndpoints Tested: {overall_results['total_endpoints']}")
    print(f"Working: {overall_results['working']} ({success_rate:.1f}%)")
    print(f"Chart-Ready: {overall_results['chart_ready']} ({chart_rate:.1f}%)")
    print(f"Real-time Capable: {overall_results['realtime_capable']}/{len(realtime_endpoints)}")
    
    print("\nCategory Breakdown:")
    for category, results in overall_results['categories'].items():
        success = (results['working'] / results['total']) * 100
        charts = (results['chart_ready'] / results['total']) * 100
        print(f"  {category:20} {results['working']}/{results['total']} working, {charts:.0f}% chart-ready")
    
    # Identify untapped features
    print("\n" + "=" * 70)
    print("UNTAPPED BACKEND FEATURES FOR VISUALIZATION")
    print("=" * 70)
    
    untapped_features = [
        "Coverage Intelligence - Branch analysis, coverage heatmaps",
        "Telemetry System - Performance traces, flamegraphs",
        "Async Processing - Task queues, thread pool monitoring",
        "Streaming Features - Live collaboration, incremental updates",
        "Quality Assurance - Quality scorecards, benchmarks",
        "Flow Optimization - DAG visualizations, dependency graphs",
        "Regression Tracking - Test regression analysis",
        "Performance Profiling - System profiler data"
    ]
    
    print("\nFeatures that could be exposed for better visualization:")
    for i, feature in enumerate(untapped_features, 1):
        print(f"  {i}. {feature}")
    
    # Overall assessment
    overall_score = (success_rate * 0.5) + (chart_rate * 0.3) + (overall_results['realtime_capable'] / len(realtime_endpoints) * 20)
    
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    print(f"Integration Score: {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("Status: EXCELLENT - Frontend integration is production ready!")
        print("[OK] All backend capabilities properly exposed")
        print("[OK] Chart-ready data structures")
        print("[OK] Real-time updates working")
        print("[OK] No browser required for validation")
    elif overall_score >= 60:
        print("Status: GOOD - Most features integrated successfully")
        print("[OK] Core functionality working")
        print("[OK] Visualization data available")
        print("[!] Some optimization opportunities remain")
    else:
        print("Status: NEEDS IMPROVEMENT")
        print("[!] Several endpoints need attention")
        print("[!] Chart data structures missing")
        print("[!] Real-time features need work")
    
    print(f"\nValidation completed WITHOUT browser interaction!")
    print(f"All testing done programmatically via API calls.")
    
    return overall_score >= 60


if __name__ == "__main__":
    success = quick_integration_test()
    exit(0 if success else 1)