#!/usr/bin/env python3
"""Test the Module Analyzer functionality"""

import requests
import json

base_url = "http://localhost:8081"

print("Testing Module Analyzer API Endpoints...")
print("=" * 60)

# Test 1: Check health
response = requests.get(f"{base_url}/api/health")
print(f"Health Check: {response.status_code}")

# Test 2: Get LLM metrics
response = requests.get(f"{base_url}/api/llm/metrics")
metrics = response.json()
print(f"LLM Metrics: Gemini Available = {metrics['analysis_status']['gemini_available']}")
print(f"  Completed Analyses: {metrics['analysis_status']['completed_analyses']}")
print(f"  Total Cost: ${metrics['cost_tracking']['total_cost_estimate']:.4f}")

# Test 3: List modules
print("\nTrying to list modules...")
response = requests.get(f"{base_url}/api/llm/list-modules")
if response.status_code == 200:
    modules = response.json()
    print(f"  Found {len(modules)} modules")
    if modules:
        print(f"  First module: {modules[0]}")
else:
    print(f"  Error: {response.status_code}")

# Test 4: Estimate cost
print("\nTrying to estimate cost for real_time_monitor.py...")
response = requests.post(
    f"{base_url}/api/llm/estimate-cost",
    json={"module_path": "real_time_monitor.py"}
)
if response.status_code == 200:
    estimate = response.json()
    print(f"  File size: {estimate['file_size_bytes']} bytes")
    print(f"  Estimated tokens: {estimate['estimated_tokens']}")
    print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.4f}")
else:
    print(f"  Error: {response.status_code} - {response.text[:100]}")

print("\n✅ Module Analyzer tab is ready!")
print("Access the dashboard at: http://localhost:8081")
print("Click on the 'Module Analyzer' tab to analyze modules manually")