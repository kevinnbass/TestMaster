"""
Check that all module files exist and have correct structure
"""

import os
from pathlib import Path

def check_modules():
    """Check module files exist and have proper structure"""
    
    base_path = Path("testmaster/analysis/comprehensive_analysis")
    
    modules = [
        # Original modules
        "supply_chain_security.py",
        "api_analysis.py", 
        "testing_analysis.py",
        "performance_analysis.py",
        "resource_io_analysis.py",
        "memory_analysis.py",
        "database_analysis.py",
        "concurrency_analysis.py",
        "error_handling_analysis.py",
        "security_analysis.py",
        # New modules
        "cognitive_load_analysis.py",
        "technical_debt_analysis.py",
        "ml_code_analysis.py"
    ]
    
    print("\n" + "="*60)
    print("MODULE FILE CHECK")
    print("="*60)
    
    existing = []
    missing = []
    
    for module in modules:
        file_path = base_path / module
        if file_path.exists():
            # Check file size to ensure it's not empty
            size = file_path.stat().st_size
            if size > 1000:  # At least 1KB
                print(f"[OK] {module} - {size:,} bytes")
                existing.append(module)
            else:
                print(f"[WARNING] {module} - File too small ({size} bytes)")
                missing.append(module)
        else:
            print(f"[MISSING] {module}")
            missing.append(module)
    
    print("\n" + "-"*60)
    print(f"SUMMARY: {len(existing)} files exist, {len(missing)} missing/incomplete")
    print("-"*60)
    
    if existing:
        print(f"\nTotal size of all modules: {sum(Path(base_path/m).stat().st_size for m in existing):,} bytes")
        print(f"Average module size: {sum(Path(base_path/m).stat().st_size for m in existing) // len(existing):,} bytes")
    
    return len(missing) == 0

if __name__ == "__main__":
    import sys
    success = check_modules()
    sys.exit(0 if success else 1)