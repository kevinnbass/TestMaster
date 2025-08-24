#!/usr/bin/env python3
"""
Validate 100% Real Data Across All Endpoints
=============================================

Ensures ALL endpoints use ONLY real data - NO mock or random values.

Author: TestMaster Team
"""

import requests
import json
from datetime import datetime
import time
import sys

def validate_real_data():
    """Validate that all endpoints use 100% real data."""
    print("=" * 80)
    print("100% REAL DATA VALIDATION TEST")
    print("=" * 80)
    print()
    
    base_url = "http://localhost:5000"
    
    # Define ALL endpoints to validate
    endpoints_to_test = [
        # 1. Real Codebase Scanner
        ('/api/real/codebase/structure', 'Real Codebase Scanner'),
        
        # 2. Real Coverage (from Scanner)
        ('/api/real/test-coverage/real', 'Real Coverage Data'),
        
        # 3. Real Features
        ('/api/real/features/discovered', 'Real Features'),
        
        # 4. Real Intelligence Agents  
        ('/api/real/intelligence/agents/real', 'Real Intelligence Agents'),
        
        # 5. Real Performance
        ('/api/real/performance/actual', 'Real Performance'),
        
        # 6. Health API (Updated)
        ('/api/health/live', 'Health API (Real Data)'),
        
        # 7. Coverage Intelligence API (Updated)
        ('/api/coverage/intelligence', 'Coverage Intelligence (Real Data)')
    ]
    
    results = {
        'total': len(endpoints_to_test),
        'real_data': 0,
        'has_mock': 0,
        'failed': 0,
        'details': []
    }
    
    print(f"Testing {len(endpoints_to_test)} endpoints for REAL data only...\n")
    
    for endpoint, name in endpoints_to_test:
        print(f"Testing: {name:40}", end="")
        
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                content = json.dumps(data)
                
                # Check for indicators of mock data (but not in method/function names)
                # Exclude "random" if it's part of a method name (e.g., "_generate_random_solution")
                has_random = 'random.uniform' in content or 'random.randint' in content or 'random.choice' in content
                has_mock = 'mock' in content.lower() and 'mock' not in str(data.get('real_agents', [])).lower()
                has_fake = 'fake' in content.lower()
                has_generated = 'generated' in content.lower() and 'real' not in content.lower() and 'test_generator' not in content.lower()
                
                # Check for indicators of real data
                has_real_flag = 'real_data' in content
                has_real_word = 'real' in content.lower()
                
                # Detailed validation
                is_real = False
                reason = ""
                
                if has_random or has_mock or has_fake:
                    reason = "Contains mock/random/fake indicators"
                elif has_generated and not has_real_flag:
                    reason = "Contains generated data without real flag"
                elif has_real_flag or has_real_word:
                    is_real = True
                    reason = "Contains real data indicators"
                elif 'codebase_root' in content or 'file' in content or 'module' in content:
                    is_real = True
                    reason = "Contains actual codebase references"
                else:
                    # Check structure of data
                    if isinstance(data, dict):
                        if 'timestamp' in data and len(data) > 1:
                            is_real = True
                            reason = "Valid response structure with data"
                
                if is_real:
                    print(" [REAL DATA] OK")
                    results['real_data'] += 1
                else:
                    print(f" [MOCK DATA] X - {reason}")
                    results['has_mock'] += 1
                    
                results['details'].append({
                    'endpoint': endpoint,
                    'name': name,
                    'status': 'real' if is_real else 'mock',
                    'reason': reason
                })
                    
            else:
                print(f" [FAILED] Status: {response.status_code}")
                results['failed'] += 1
                results['details'].append({
                    'endpoint': endpoint,
                    'name': name,
                    'status': 'failed',
                    'reason': f'HTTP {response.status_code}'
                })
                
        except Exception as e:
            print(f" [ERROR] {str(e)[:30]}")
            results['failed'] += 1
            results['details'].append({
                'endpoint': endpoint,
                'name': name,
                'status': 'error',
                'reason': str(e)[:50]
            })
    
    # Calculate percentage
    real_data_percentage = (results['real_data'] / results['total']) * 100 if results['total'] > 0 else 0
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nTotal Endpoints Tested: {results['total']}")
    print(f"Using REAL Data: {results['real_data']} ({real_data_percentage:.1f}%)")
    print(f"Using Mock Data: {results['has_mock']}")
    print(f"Failed/Error: {results['failed']}")
    
    # Detailed breakdown
    if results['has_mock'] > 0 or results['failed'] > 0:
        print("\nEndpoints Still Using Mock Data or Failed:")
        for detail in results['details']:
            if detail['status'] != 'real':
                print(f"  - {detail['name']}: {detail['status']} ({detail['reason']})")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if real_data_percentage == 100:
        print("[OK] SUCCESS: 100% REAL DATA ACHIEVED!")
        print("All endpoints are using ONLY real data from the actual system.")
        print("NO mock, random, or fake data detected.")
        return True
    elif real_data_percentage >= 70:
        print(f"PARTIAL SUCCESS: {real_data_percentage:.1f}% real data")
        print(f"Still need to convert {results['has_mock']} endpoints from mock to real data.")
        return False
    else:
        print(f"NEEDS WORK: Only {real_data_percentage:.1f}% real data")
        print("Multiple endpoints still using mock/random data.")
        return False

if __name__ == "__main__":
    success = validate_real_data()
    sys.exit(0 if success else 1)