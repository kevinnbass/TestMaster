import json
import os
from datetime import datetime

scan_file = os.path.join("tools", "codebase_monitor", "reports", "latest", "scan.json")
output_file = os.path.join("tools", "codebase_monitor", "outputs", "duplicate_groups_topN.json")

if not os.path.exists(scan_file):
    print(f"Scan file not found: {scan_file}")
    exit(1)

try:
    with open(scan_file, 'r', encoding='utf-8') as f:
        scan = json.load(f)
    
    duplicates = scan.get('duplicates', [])
    top_groups = duplicates[:200]  # Take first 200 groups
    
    output_data = {
        "duplicate_groups": top_groups,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_groups": len(top_groups),
        "source_scan": scan_file,
        "note": f"Top {len(top_groups)} duplicate groups from {len(duplicates)} total groups"
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Generated duplicate groups: {len(top_groups)} groups out of {len(duplicates)} total")
    print(f"Saved to: {output_file}")
    
except Exception as e:
    print(f"Failed to process scan file: {e}")
    exit(1)