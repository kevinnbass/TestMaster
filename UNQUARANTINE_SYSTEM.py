#!/usr/bin/env python3
"""
Safe Unquarantine System
Restores quarantined files with validation
"""

import os
import shutil
from pathlib import Path
import json
import logging

class SafeUnquarantineSystem:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.quarantine_dir = self.base_path / "QUARANTINE"
        self.testmaster_dir = self.base_path / "TestMaster"
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_original_path(self, quarantined_file: str) -> str:
        """Determine original path for quarantined file"""
        # Remove timestamp suffix to get original name
        parts = quarantined_file.split('_')
        if len(parts) >= 3 and parts[-2].startswith('202'):
            # Remove timestamp parts (date and time)
            original_name = '_'.join(parts[:-2]) + '.py'
        else:
            original_name = quarantined_file
        
        return str(self.testmaster_dir / original_name)

    def restore_file(self, quarantined_file: Path) -> bool:
        """Safely restore a single quarantined file"""
        try:
            # Determine original location
            original_path = self.get_original_path(quarantined_file.name)
            original_file = Path(original_path)
            
            # Create directory if needed
            original_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file back (don't overwrite if exists)
            if not original_file.exists():
                shutil.copy2(quarantined_file, original_file)
                self.logger.info(f"Restored: {quarantined_file.name} -> {original_file}")
                return True
            else:
                self.logger.warning(f"File already exists, skipping: {original_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore {quarantined_file}: {e}")
            return False

    def restore_all_files(self) -> dict:
        """Restore all quarantined files"""
        results = {
            'restored': [],
            'skipped': [],
            'failed': []
        }
        
        if not self.quarantine_dir.exists():
            self.logger.warning("No quarantine directory found")
            return results
        
        for quarantined_file in self.quarantine_dir.glob("*.py"):
            try:
                success = self.restore_file(quarantined_file)
                if success:
                    results['restored'].append(str(quarantined_file))
                else:
                    results['skipped'].append(str(quarantined_file))
            except Exception as e:
                results['failed'].append(str(quarantined_file))
                self.logger.error(f"Failed to restore {quarantined_file}: {e}")
        
        return results

    def generate_restore_report(self, results: dict):
        """Generate restore operation report"""
        report = {
            'timestamp': str(datetime.datetime.now()),
            'operation': 'unquarantine',
            'summary': {
                'total_files': len(results['restored']) + len(results['skipped']) + len(results['failed']),
                'files_restored': len(results['restored']),
                'files_skipped': len(results['skipped']),
                'files_failed': len(results['failed'])
            },
            'details': results
        }
        
        report_file = self.base_path / "UNQUARANTINE_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report

def main():
    """Execute unquarantine operation"""
    print("TestMaster Safe Unquarantine System")
    print("=" * 40)
    
    unquarantine = SafeUnquarantineSystem()
    
    print("Restoring quarantined files...")
    results = unquarantine.restore_all_files()
    
    print(f"\\nRestore Results:")
    print(f"- Files restored: {len(results['restored'])}")
    print(f"- Files skipped (already exist): {len(results['skipped'])}")
    print(f"- Files failed: {len(results['failed'])}")
    
    if results['restored']:
        print("\\nRestored files:")
        for file in results['restored'][:10]:  # Show first 10
            print(f"  - {Path(file).name}")
        if len(results['restored']) > 10:
            print(f"  ... and {len(results['restored']) - 10} more")
    
    # Generate report
    report = unquarantine.generate_restore_report(results)
    print(f"\\nDetailed report saved to: UNQUARANTINE_REPORT.json")
    
    return results

if __name__ == "__main__":
    import datetime
    main()