#!/usr/bin/env python3
"""
Simple baseline creation and improvement validation script.
"""

import json
import os
from pathlib import Path

def main():
    # Set up paths
    root = Path(__file__).parent.parent.parent
    latest_scan = root / "tools" / "codebase_monitor" / "reports" / "latest" / "scan.json"
    baseline = root / "tools" / "codebase_monitor" / "outputs" / "refactor_baseline.json"
    
    # Load latest scan
    if not latest_scan.exists():
        print("Error: No latest scan found")
        return 1
    
    with open(latest_scan, 'r', encoding='utf-8') as f:
        scan_data = json.load(f)
    
    # Create baseline metrics
    baseline_data = {
        "timestamp": scan_data.get("generated_at", "unknown"),
        "total_files": scan_data.get("total_files", 0),
        "total_code_lines": scan_data.get("total_code_lines", 0),
        "duplicate_groups": len(scan_data.get("duplicates", [])),
        "hotspots": {}
    }
    
    # Extract hotspot counts
    for hotspot_type, files in scan_data.get("hotspots", {}).items():
        baseline_data["hotspots"][hotspot_type] = len(files) if isinstance(files, list) else 0
    
    # Save baseline
    os.makedirs(baseline.parent, exist_ok=True)
    with open(baseline, 'w', encoding='utf-8') as f:
        json.dump(baseline_data, f, indent=2)
    
    print(f"Baseline created successfully: {baseline}")
    print(f"  Total files: {baseline_data['total_files']}")
    print(f"  Total lines: {baseline_data['total_code_lines']}")
    print(f"  Duplicate groups: {baseline_data['duplicate_groups']}")
    print(f"  Hotspot categories: {len(baseline_data['hotspots'])}")
    
    # Show top hotspot categories
    hotspot_counts = [(k, v) for k, v in baseline_data["hotspots"].items()]
    hotspot_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop hotspot categories:")
    for hotspot_type, count in hotspot_counts[:5]:
        print(f"  {hotspot_type}: {count} files")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())