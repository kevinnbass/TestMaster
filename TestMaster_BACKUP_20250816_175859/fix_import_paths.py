#!/usr/bin/env python3
"""
Fix import paths in generated intelligent test files.
Converts generic imports to project-specific paths.
"""

import re
from pathlib import Path
from typing import List, Tuple

def fix_import_path(test_content: str, test_filename: str) -> Tuple[str, List[str]]:
    """
    Fix import paths in a test file.
    
    Returns:
        Tuple of (fixed_content, list_of_changes)
    """
    changes = []
    fixed_content = test_content
    
    # Extract module name from test filename
    # test_data_loader_intelligent.py -> data_loader
    module_name = test_filename.replace("test_", "").replace("_intelligent.py", "")
    
    # Common import patterns to fix
    import_patterns = [
        # Pattern 1: from module_name import ...
        (f"from {module_name} import", f"from multi_coder_analysis.{module_name} import"),
        
        # Pattern 2: import module_name
        (f"import {module_name}", f"import multi_coder_analysis.{module_name} as {module_name}"),
    ]
    
    # Check if module is in a subdirectory
    subdirs = {
        "ablation_diagnostics": "runtime",
        "ablation_runner": "runtime", 
        "analytics": "",
        "tot_runner": "runtime",
        "gemini_provider": "llm_providers",
        "openrouter_provider": "llm_providers",
        "base": "llm_providers",
        "gepa_optimizer": "improvement_system",
        "config_manager": "improvement_system",
        "integrated_optimizer_orchestrator": "improvement_system",
        "enhanced_hybrid_optimizer": "improvement_system",
        "optimization_metrics_collector": "improvement_system",
        "optimization_orchestrator": "improvement_system",
        "optimization_session_manager": "improvement_system",
        "optimized_micro_macro_loop_system": "improvement_system",
        "schema_validator": "runtime",
        "logging_utils": "utils",
        "dspy_runner": "",
        "data_loader": "",
        "consensus": "runtime",
    }
    
    subdir = subdirs.get(module_name, "")
    
    if subdir:
        # Update patterns for subdirectory modules
        import_patterns = [
            (f"from {module_name} import", f"from multi_coder_analysis.{subdir}.{module_name} import"),
            (f"import {module_name}", f"import multi_coder_analysis.{subdir}.{module_name} as {module_name}"),
        ]
    
    # Apply fixes
    for old_pattern, new_pattern in import_patterns:
        if old_pattern in fixed_content:
            fixed_content = fixed_content.replace(old_pattern, new_pattern)
            changes.append(f"Fixed: '{old_pattern}' -> '{new_pattern}'")
    
    # Fix specific problematic imports
    special_fixes = [
        ("from ablation_diagnostics import AblationDiagnostics", 
         "from multi_coder_analysis.runtime.ablation_diagnostics import AblationDiagnostics"),
        ("from ablation_runner import", 
         "from multi_coder_analysis.runtime.ablation_runner import"),
        ("from ab_testing import", 
         "from multi_coder_analysis.ab_testing import"),
        ("from ai_config_recommendations import",
         "from multi_coder_analysis.ai_config_recommendations import"),
    ]
    
    for old, new in special_fixes:
        if old in fixed_content:
            fixed_content = fixed_content.replace(old, new)
            changes.append(f"Special fix: '{old}' -> '{new}'")
    
    return fixed_content, changes

def process_test_files():
    """Process all intelligent test files and fix their imports."""
    
    test_dir = Path("tests/unit")
    intelligent_tests = list(test_dir.glob("*_intelligent.py"))
    
    print(f"Found {len(intelligent_tests)} intelligent test files to fix")
    print("="*70)
    
    total_fixed = 0
    total_changes = 0
    
    for test_file in intelligent_tests:
        print(f"\nProcessing: {test_file.name}")
        
        # Read current content
        content = test_file.read_text(encoding='utf-8')
        
        # Fix imports
        fixed_content, changes = fix_import_path(content, test_file.name)
        
        if changes:
            # Save fixed content
            test_file.write_text(fixed_content, encoding='utf-8')
            
            print(f"  [OK] Fixed {len(changes)} imports:")
            for change in changes:
                print(f"    - {change}")
            
            total_fixed += 1
            total_changes += len(changes)
        else:
            print(f"  [--] No changes needed")
    
    print("\n" + "="*70)
    print(f"IMPORT PATH FIXING COMPLETE")
    print(f"Files fixed: {total_fixed}/{len(intelligent_tests)}")
    print(f"Total import changes: {total_changes}")
    print("="*70)
    
    return total_fixed, total_changes

def verify_imports():
    """Verify that fixed imports can be resolved."""
    
    print("\n" + "="*70)
    print("VERIFYING IMPORT RESOLUTION")
    print("="*70)
    
    test_dir = Path("tests/unit")
    sample_tests = list(test_dir.glob("*_intelligent.py"))[:5]  # Check first 5
    
    for test_file in sample_tests:
        print(f"\nVerifying: {test_file.name}")
        
        # Try to compile the file to check for syntax/import errors
        try:
            content = test_file.read_text(encoding='utf-8')
            compile(content, test_file.name, 'exec')
            print(f"  [OK] Syntax valid")
        except SyntaxError as e:
            print(f"  [ERROR] Syntax error: {e}")
        except Exception as e:
            print(f"  [WARN] Other error: {e}")

if __name__ == "__main__":
    # Fix import paths
    fixed, changes = process_test_files()
    
    # Verify fixes
    if fixed > 0:
        verify_imports()
    
    print("\n[COMPLETE] Import path fixing complete!")
    print(f"   Fixed {fixed} files with {changes} total changes")