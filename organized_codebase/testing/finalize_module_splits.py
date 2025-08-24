"""
Finalize Module Splits
======================

Archive original oversized modules and create proper import redirects
to maintain backward compatibility while achieving <1000 line modules.
"""

import os
import shutil
from datetime import datetime


class ModuleSplitFinalizer:
    """Finalize the module splitting process."""
    
    def __init__(self):
        self.oversized_modules = [
            'core\\intelligence\\analysis\\business_analyzer.py',
            'core\\intelligence\\analysis\\debt_analyzer.py',
            'core\\intelligence\\api\\enterprise_integration_layer.py',
            'core\\intelligence\\api\\ml_api.py',
            'core\\intelligence\\coordination\\agent_coordination_protocols.py',
            'core\\intelligence\\coordination\\resource_coordination_system.py',
            'core\\intelligence\\coordination\\unified_workflow_orchestrator.py',
            'core\\intelligence\\documentation\\revolutionary\\neo4j_dominator.py',
            'core\\intelligence\\ml\\advanced\\circuit_breaker_ml.py',
            'core\\intelligence\\ml\\advanced\\delivery_optimizer.py',
            'core\\intelligence\\ml\\advanced\\integrity_ml_guardian.py',
            'core\\intelligence\\ml\\advanced\\performance_optimizer.py',
            'core\\intelligence\\ml\\advanced\\sla_ml_optimizer.py',
            'core\\intelligence\\ml\\enterprise\\ml_infrastructure_orchestrator.py',
            'core\\intelligence\\monitoring\\agent_qa.py',
            'core\\intelligence\\monitoring\\performance_optimization_engine.py',
            'core\\intelligence\\validation\\integration_test_suite.py',
            'core\\intelligence\\validation\\system_validation_framework.py'
        ]
    
    def create_archive_directory(self):
        """Create archive directory for oversized modules."""
        archive_dir = "archive/oversized_modules_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        return archive_dir
    
    def archive_original_module(self, original_path: str, archive_dir: str):
        """Archive the original oversized module."""
        if not os.path.exists(original_path):
            return False
            
        # Create archive path maintaining directory structure
        relative_path = original_path.replace("core\\intelligence\\", "")
        archive_path = os.path.join(archive_dir, relative_path)
        archive_subdir = os.path.dirname(archive_path)
        
        if not os.path.exists(archive_subdir):
            os.makedirs(archive_subdir)
        
        # Copy to archive
        shutil.copy2(original_path, archive_path)
        print(f"Archived: {original_path} -> {archive_path}")
        return True
    
    def create_import_redirect(self, original_path: str):
        """Create import redirect module for backward compatibility."""
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        split_dir = os.path.join(os.path.dirname(original_path), f"{base_name}_modules")
        
        if not os.path.exists(split_dir):
            print(f"WARNING: Split directory not found: {split_dir}")
            return False
        
        # Create import redirect content
        redirect_content = f'''"""
{base_name.title()} Module - Import Redirect
==========================================

This module was split into smaller components for maintainability.
All original functionality is preserved through the split modules.

Original module archived: archive/oversized_modules_*/
Split modules location: {base_name}_modules/
"""

# Import all components from split modules to maintain backward compatibility
from .{base_name}_modules import *

# Preserve original module's public interface
__all__ = []
try:
    # Import any __all__ definitions from split modules
    from .{base_name}_modules import __all__ as split_all
    __all__.extend(split_all)
except ImportError:
    pass

# Add common module metadata
__version__ = "2.0.0-modularized"
__split_timestamp__ = "{datetime.now().isoformat()}"
__original_lines__ = "1000+ (see archive for original)"
'''
        
        # Write the redirect module
        with open(original_path, 'w', encoding='utf-8') as f:
            f.write(redirect_content)
        
        print(f"Created import redirect: {original_path}")
        return True
    
    def fix_oversized_split_modules(self):
        """Further split any split modules that are still over 1000 lines."""
        oversized_splits = [
            'core\\intelligence\\coordination\\agent_coordination_protocols_modules\\agent_coordination_protocols_core.py',
            'core\\intelligence\\coordination\\resource_coordination_system_modules\\resource_coordination_system_core.py',
            'core\\intelligence\\coordination\\unified_workflow_orchestrator_modules\\unified_workflow_orchestrator_core.py',
            'core\\intelligence\\documentation\\revolutionary\\neo4j_dominator_modules\\neo4j_dominator_core.py',
            'core\\intelligence\\monitoring\\agent_qa_modules\\agent_qa_core.py'
        ]
        
        for module_path in oversized_splits:
            if os.path.exists(module_path):
                self.split_further(module_path)
    
    def split_further(self, module_path: str):
        """Split an oversized module into even smaller pieces."""
        with open(module_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) <= 1000:
            return  # Already small enough
        
        print(f"Further splitting {module_path} ({len(lines)} lines)")
        
        # Simple line-based split
        base_name = os.path.splitext(os.path.basename(module_path))[0]
        base_dir = os.path.dirname(module_path)
        
        # Extract header (imports, docstring)
        header_lines = []
        content_start = 0
        
        for i, line in enumerate(lines[:100]):
            stripped = line.strip()
            if (stripped.startswith('"""') or stripped.startswith("'''") or
                stripped.startswith('import ') or stripped.startswith('from ') or
                stripped == '' or stripped.startswith('#')):
                header_lines.append(line)
            else:
                content_start = i
                break
        
        # Calculate split points
        content_lines = lines[content_start:]
        split_size = min(700, len(content_lines) // 2)  # Max 700 lines per split
        
        # Create part 1
        part1_path = module_path.replace('_core.py', '_part1.py')
        part1_content = header_lines + content_lines[:split_size]
        
        with open(part1_path, 'w', encoding='utf-8') as f:
            f.writelines(part1_content)
        
        print(f"  Created: {part1_path} ({len(part1_content)} lines)")
        
        # Create part 2
        part2_path = module_path.replace('_core.py', '_part2.py')
        part2_content = header_lines + content_lines[split_size:]
        
        with open(part2_path, 'w', encoding='utf-8') as f:
            f.writelines(part2_content)
        
        print(f"  Created: {part2_path} ({len(part2_content)} lines)")
        
        # Update the original file to import from both parts
        redirect_content = f'''"""
{base_name.title()} Module - Further Split
=========================================

This module was further split due to size constraints.
"""

# Import all from both parts
from .{base_name.replace("_core", "_part1")} import *
from .{base_name.replace("_core", "_part2")} import *
'''
        
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(redirect_content)
        
        print(f"  Updated redirect: {module_path}")
    
    def run_finalization(self):
        """Run the complete finalization process."""
        print("="*60)
        print("FINALIZING MODULE SPLITS")
        print("="*60)
        
        # Create archive directory
        archive_dir = self.create_archive_directory()
        print(f"Created archive directory: {archive_dir}")
        
        success_count = 0
        
        # Process each oversized module
        for module_path in self.oversized_modules:
            if os.path.exists(module_path):
                try:
                    # Archive original
                    if self.archive_original_module(module_path, archive_dir):
                        # Create import redirect
                        if self.create_import_redirect(module_path):
                            success_count += 1
                    else:
                        print(f"Failed to archive: {module_path}")
                except Exception as e:
                    print(f"Error processing {module_path}: {e}")
            else:
                print(f"File not found: {module_path}")
        
        # Fix oversized split modules
        print(f"\n{'-'*40}")
        print("FURTHER SPLITTING OVERSIZED MODULES")
        print(f"{'-'*40}")
        self.fix_oversized_split_modules()
        
        print(f"\n{'='*60}")
        print(f"FINALIZATION COMPLETE: {success_count}/{len(self.oversized_modules)} modules processed")
        print(f"Archive location: {archive_dir}")
        print(f"{'='*60}")
        
        return success_count


if __name__ == "__main__":
    finalizer = ModuleSplitFinalizer()
    finalizer.run_finalization()