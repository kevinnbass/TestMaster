#!/usr/bin/env python3
"""
Quality gate enforcement system for codebase monitoring.

This script enforces quality gates to prevent regression in code metrics
by comparing current scan results with a baseline.
"""

import json
import os
import sys
import yaml
from pathlib import Path

# Configuration paths
ROOT = Path(__file__).parent.parent.parent  # testmaster root
CFG = ROOT / "tools" / "codebase_monitor" / "config.yml"
NEW = ROOT / "tools" / "codebase_monitor" / "reports" / "latest" / "scan.json"
BASELINE = ROOT / "tools" / "codebase_monitor" / "outputs" / "baseline_scan.json"


def load_json(path):
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {path}: {e}")
        return None


def sum_hotspots(data, key):
    """Count total files in a hotspot category."""
    if not data or "hotspots" not in data:
        return 0
    return len(data.get("hotspots", {}).get(key, []))


def get_total_files(data):
    """Get total file count from scan data."""
    return data.get("total_files", 0) if data else 0


def get_total_size_bytes(data):
    """Get total codebase size in bytes."""
    return data.get("total_size_bytes", 0) if data else 0


def check_file_size_gate(cfg, new_data, baseline_data):
    """Check if new large files have been introduced."""
    failures = []
    
    if cfg["quality_gate"].get("block_new_large_binaries_outside_artifacts", False):
        new_large = sum_hotspots(new_data, "large_files") 
        baseline_large = sum_hotspots(baseline_data, "large_files")
        
        if new_large > baseline_large:
            failures.append(f"Large files increased from {baseline_large} to {new_large}")
    
    return failures


def check_ts_any_gate(cfg, new_data, baseline_data):
    """Check if TypeScript 'any' usage has increased."""
    failures = []
    
    if cfg["quality_gate"].get("block_ts_any_increase", False):
        new_any = sum_hotspots(new_data, "ts_any_overuse")
        baseline_any = sum_hotspots(baseline_data, "ts_any_overuse")
        
        if new_any > baseline_any:
            failures.append(f"TypeScript 'any' overuse increased from {baseline_any} to {new_any}")
    
    return failures


def check_eslint_disable_gate(cfg, new_data, baseline_data):
    """Check if ESLint disable comments have increased."""
    failures = []
    
    if cfg["quality_gate"].get("block_eslint_disable_increase", False):
        new_ignored = sum_hotspots(new_data, "eslint_ts_ignored_heavy")
        baseline_ignored = sum_hotspots(baseline_data, "eslint_ts_ignored_heavy")
        
        if new_ignored > baseline_ignored:
            failures.append(f"ESLint/TS-ignore usage increased from {baseline_ignored} to {new_ignored}")
    
    return failures


def check_python_complexity_gate(cfg, new_data, baseline_data):
    """Check if Python complexity has increased."""
    failures = []
    
    if cfg["quality_gate"].get("block_py_complexity_increase", False):
        new_complex = sum_hotspots(new_data, "high_branching_python")
        baseline_complex = sum_hotspots(baseline_data, "high_branching_python")
        
        if new_complex > baseline_complex:
            failures.append(f"Python complexity increased from {baseline_complex} to {new_complex}")
    
    return failures


def create_baseline_if_missing(new_data):
    """Create baseline from current scan if it doesn't exist."""
    try:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2)
        print(f"[OK] Baseline created at: {BASELINE}")
        print(f"  Total files: {get_total_files(new_data)}")
        print(f"  Total size: {get_total_size_bytes(new_data):,} bytes")
        return True
    except Exception as e:
        print(f"Error creating baseline: {e}")
        return False


def compare_metrics(new_data, baseline_data):
    """Compare current scan with baseline and return improvement metrics."""
    metrics = {}
    
    # File counts
    metrics["total_files"] = {
        "new": get_total_files(new_data),
        "baseline": get_total_files(baseline_data),
        "change": get_total_files(new_data) - get_total_files(baseline_data)
    }
    
    # Size metrics  
    metrics["total_size_bytes"] = {
        "new": get_total_size_bytes(new_data),
        "baseline": get_total_size_bytes(baseline_data),
        "change": get_total_size_bytes(new_data) - get_total_size_bytes(baseline_data)
    }
    
    # Hotspot comparisons
    hotspot_types = [
        "large_files",
        "ts_any_overuse", 
        "eslint_ts_ignored_heavy",
        "high_branching_python",
        "long_python_functions",
        "mixed_indentation"
    ]
    
    for hotspot_type in hotspot_types:
        new_count = sum_hotspots(new_data, hotspot_type)
        baseline_count = sum_hotspots(baseline_data, hotspot_type)
        metrics[hotspot_type] = {
            "new": new_count,
            "baseline": baseline_count, 
            "change": new_count - baseline_count
        }
    
    return metrics


def main():
    """Main quality gate enforcement logic."""
    
    # Load configuration
    if not CFG.exists():
        print(f"Error: Configuration file not found: {CFG}")
        return 1
    
    try:
        with open(CFG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Load new scan results
    new_data = load_json(NEW)
    if not new_data:
        print(f"Error: Could not load current scan results from: {NEW}")
        print("Please run the analyzer first to generate a scan report.")
        return 1
    
    # Check if baseline exists, create if not
    if not BASELINE.exists():
        print("No baseline found, creating from current scan...")
        if create_baseline_if_missing(new_data):
            print("[OK] Quality gate: BASELINE_CREATED")
            return 0
        else:
            return 1
    
    # Load baseline
    baseline_data = load_json(BASELINE)
    if not baseline_data:
        print(f"Error: Could not load baseline data from: {BASELINE}")
        return 1
    
    # Run all quality gate checks
    failures = []
    
    failures.extend(check_file_size_gate(cfg, new_data, baseline_data))
    failures.extend(check_ts_any_gate(cfg, new_data, baseline_data))
    failures.extend(check_eslint_disable_gate(cfg, new_data, baseline_data))
    failures.extend(check_python_complexity_gate(cfg, new_data, baseline_data))
    
    # Compare metrics for reporting
    metrics = compare_metrics(new_data, baseline_data)
    
    # Generate results
    result = {
        "gate_status": "fail" if failures else "pass",
        "timestamp": new_data.get("generated_at", "unknown"),
        "baseline_timestamp": baseline_data.get("generated_at", "unknown"),
        "failures": failures,
        "metrics_comparison": metrics
    }
    
    # Output results
    if failures:
        print("[FAIL] Quality gate: FAIL")
        print("Reasons:")
        for failure in failures:
            print(f"  - {failure}")
        print(json.dumps(result, indent=2))
        return 2
    else:
        print("[PASS] Quality gate: PASS")
        
        # Show improvements if any
        improvements = []
        for metric_name, metric_data in metrics.items():
            if metric_data["change"] < 0:  # Negative change is good for most metrics
                improvements.append(f"{metric_name}: {metric_data['change']}")
        
        if improvements:
            print("Improvements detected:")
            for improvement in improvements[:5]:  # Show top 5
                print(f"  + {improvement}")
        
        print(json.dumps(result, indent=2))
        return 0


if __name__ == "__main__":
    sys.exit(main())