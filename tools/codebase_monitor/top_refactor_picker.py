import json
import os
import sys


def main() -> int:
    """Generate a top-100 file list to refactor based on hotspot frequency.

    Reads tools\codebase_monitor\reports\latest\scan.json and writes
    tools\codebase_monitor\outputs\refactor_top100.json
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    scan_path = os.path.join(repo_root, "tools", "codebase_monitor", "reports", "latest", "scan.json")
    out_path = os.path.join(repo_root, "tools", "codebase_monitor", "outputs", "refactor_top100.json")

    if not os.path.isfile(scan_path):
        print(f"Scan JSON not found: {scan_path}")
        return 2

    with open(scan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    freq = {}
    hotspots = data.get("hotspots", {}) or {}
    for paths in hotspots.values():
        for rel_path in paths:
            freq[rel_path] = freq.get(rel_path, 0) + 1

    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:100]
    out = {"top100": [p for p, _ in top]}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

import json
import os
import sys
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent.parent  # testmaster root
REPORT = ROOT / "tools" / "codebase_monitor" / "reports" / "latest" / "scan.json"
FALLBACK_REPORTS = ROOT / "tools" / "codebase_monitor" / "reports"
OUT = ROOT / "tools" / "codebase_monitor" / "outputs" / "refactor_top100.json"

def score_file(f):
    """
    Score a file based on refactoring priority.
    Higher scores indicate higher refactoring priority.
    """
    s = 0
    
    # Branch complexity is a strong indicator of refactoring need
    s += f.get("py_branch_nodes", 0) * 3
    
    # Long functions are harder to maintain
    s += int(f.get("py_avg_function_len", 0) * 2)
    
    # Very large files need breaking up
    if f.get("num_lines", 0) >= 1000: 
        s += 50
    elif f.get("num_lines", 0) >= 500:
        s += 25
    
    # Mixed indentation is a code smell
    if f.get("has_mixed_indent", False): 
        s += 10
    
    # TODOs indicate technical debt
    s += f.get("num_todos", 0) * 2
    
    return s

def find_latest_scan():
    """Find the latest scan report if 'latest' symlink doesn't exist."""
    if REPORT.exists():
        return REPORT
    
    # Look for most recent scan_*.json file
    if FALLBACK_REPORTS.exists():
        scan_files = sorted(FALLBACK_REPORTS.glob("scan_*.json"), 
                          key=lambda p: p.stat().st_mtime, 
                          reverse=True)
        if scan_files:
            return scan_files[0]
    
    return None

def main():
    # Find scan report
    scan_path = find_latest_scan()
    if not scan_path or not scan_path.exists():
        print(f"Error: No scan report found at {REPORT}")
        print("Please run the analyzer first:")
        print("  python .\\tools\\codebase_monitor\\analyzer.py --root . --output-dir .\\tools\\codebase_monitor\\reports")
        return 1
    
    print(f"Using scan report: {scan_path}")
    
    # Load scan data
    with open(scan_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract files to consider for refactoring
    candidates = []
    
    # If we have detailed file metrics, use them
    if "file_metrics" in data:
        for fm in data["file_metrics"]:
            # Only consider Python files for now
            if fm.get("extension") in [".py", ".pyx"]:
                fm["score"] = score_file(fm)
                candidates.append(fm)
    
    # Fallback: use hotspot data to identify problematic files
    if not candidates and "hotspots" in data:
        print("No detailed file metrics found, using hotspot analysis...")
        
        # Count how many hotspot categories each file appears in
        freq = {}
        hotspot_types = {}
        
        for hotspot_type, file_list in data.get("hotspots", {}).items():
            for filepath in file_list:
                freq[filepath] = freq.get(filepath, 0) + 1
                if filepath not in hotspot_types:
                    hotspot_types[filepath] = []
                hotspot_types[filepath].append(hotspot_type)
        
        # Create candidate entries from hotspot frequency
        for filepath, count in freq.items():
            # Only consider Python files
            if filepath.endswith((".py", ".pyx")):
                candidates.append({
                    "rel_path": filepath,
                    "hotspot_count": count,
                    "hotspot_types": hotspot_types[filepath],
                    "score": count * 10  # Simple scoring based on hotspot frequency
                })
    
    # Sort by score (descending) and take top 100
    candidates.sort(key=lambda x: (-x.get("score", 0), x.get("rel_path", "")))
    top100 = candidates[:100]
    
    # Extract just the paths for the output
    output_data = {
        "top100": [c.get("rel_path") for c in top100],
        "detailed": [  # Include detailed info for debugging
            {
                "path": c.get("rel_path"),
                "score": c.get("score", 0),
                "lines": c.get("num_lines", 0),
                "todos": c.get("num_todos", 0),
                "branch_nodes": c.get("py_branch_nodes", 0),
                "avg_func_len": c.get("py_avg_function_len", 0),
                "mixed_indent": c.get("has_mixed_indent", False),
                "hotspot_count": c.get("hotspot_count", 0),
                "hotspot_types": c.get("hotspot_types", [])
            } for c in top100
        ],
        "total_candidates": len(candidates),
        "scan_timestamp": data.get("generated_at", "unknown")
    }
    
    # Ensure output directory exists
    OUT.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Wrote top 100 refactoring targets to: {OUT}")
    print(f"Total candidates considered: {len(candidates)}")
    if top100:
        print(f"Top file: {top100[0].get('rel_path')} (score: {top100[0].get('score', 0)})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())