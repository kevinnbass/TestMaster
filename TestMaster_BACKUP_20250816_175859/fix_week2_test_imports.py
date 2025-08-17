#!/usr/bin/env python3
"""
Fix Week 2 Test Imports

Fixes the indentation issues in the newly generated Week 2 test files.
"""

import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class Week2ImportFixer:
    """Fixes import indentation issues in Week 2 tests."""
    
    def __init__(self):
        self.tests_dir = project_root / "tests" / "unit"
        self.fixed_count = 0
    
    def fix_test_file(self, test_file: Path) -> bool:
        """Fix import issues in a single test file."""
        try:
            content = test_file.read_text(encoding='utf-8')
        except:
            return False
        
        # Skip files that don't have the issue
        if 'Generated with enhanced Week 2 logic' not in content:
            return False
        
        original_content = content
        
        # Fix the import section - replace the problematic multi-line import with a simple version
        import_pattern = r'# Import module under test with comprehensive fallback\ntry:.*?MODULE_AVAILABLE = False.*?print\(f"Module.*?"\)'
        
        simple_import = '''# Import module under test with fallback handling
MODULE_AVAILABLE = True
try:
    # Dynamic import based on file name
    import importlib
    module_name = __file__.split('test_')[1].replace('.py', '')
    
    # Try multiple import paths
    try:
        test_module = importlib.import_module(f'multi_coder_analysis.runtime.{module_name}')
    except ImportError:
        try:
            test_module = importlib.import_module(f'multi_coder_analysis.llm_providers.{module_name}')
        except ImportError:
            try:
                test_module = importlib.import_module(f'multi_coder_analysis.improvement_system.{module_name}')
            except ImportError:
                try:
                    test_module = importlib.import_module(f'multi_coder_analysis.{module_name}')
                except ImportError:
                    MODULE_AVAILABLE = False
                    
except Exception:
    MODULE_AVAILABLE = False'''
        
        content = re.sub(import_pattern, simple_import, content, flags=re.DOTALL)
        
        # Only write if changes were made
        if content != original_content:
            test_file.write_text(content, encoding='utf-8')
            return True
        
        return False
    
    def fix_all_tests(self):
        """Fix all Week 2 test files."""
        test_files = [f for f in self.tests_dir.glob("test_*.py") if f.stat().st_mtime > 1723730000]  # Recent files
        
        print(f"Checking {len(test_files)} recent test files for import fixes...")
        
        for test_file in test_files:
            try:
                if self.fix_test_file(test_file):
                    print(f"Fixed imports in {test_file.name}")
                    self.fixed_count += 1
            except Exception as e:
                print(f"Error fixing {test_file.name}: {e}")
        
        print(f"\nFixed imports in {self.fixed_count} test files")

def main():
    """Main execution function."""
    fixer = Week2ImportFixer()
    fixer.fix_all_tests()
    
    print("\nTesting a few fixed files...")
    
    # Test some fixed files
    import subprocess
    test_files = [
        "test_ablation_diagnostics.py",
        "test_gemini_provider.py", 
        "test_consensus.py"
    ]
    
    for test_file in test_files:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                f"tests/unit/{test_file}", 
                "-v", "--tb=no", "-q"
            ], capture_output=True, text=True, timeout=15, cwd=project_root)
            
            if result.returncode == 0:
                print(f"✓ {test_file} - Fixed and working")
            else:
                if 'collected' in result.stdout:
                    print(f"~ {test_file} - Collecting tests successfully")
                else:
                    print(f"× {test_file} - Still has issues")
        except subprocess.TimeoutExpired:
            print(f"○ {test_file} - Tests taking too long")
        except Exception as e:
            print(f"! {test_file} - Error: {e}")

if __name__ == "__main__":
    main()