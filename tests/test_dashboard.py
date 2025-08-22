#!/usr/bin/env python3
"""
Quick test to check if dashboard is running
"""

import requests
import json

try:
    # Test the dashboard API
    response = requests.get('http://localhost:8080/api/status', timeout=5)
    if response.status_code == 200:
        data = response.json()
        print("[OK] Dashboard is running!")
        print(f"Status: {json.dumps(data, indent=2)}")
        
        # Test summary endpoint
        response = requests.get('http://localhost:8080/api/summary', timeout=5)
        if response.status_code == 200:
            summary = response.json()
            print(f"Summary: {json.dumps(summary, indent=2)}")
    else:
        print(f"[ERROR] Dashboard responded with status {response.status_code}")

except requests.exceptions.ConnectionError:
    print("[ERROR] Dashboard not running or not accessible at localhost:8080")
except Exception as e:
    print(f"[ERROR] {e}")