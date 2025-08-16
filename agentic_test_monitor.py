#!/usr/bin/env python3
"""
Agentic Test Monitoring System
Tracks code changes, follows refactorings, and maintains test suite automatically.

Features:
1. Monitors file changes via git
2. Detects module splits/merges/renames
3. Updates test imports automatically
4. Generates tests for new modules
5. Runs after coding sessions or on schedule
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional
import ast
import difflib
import schedule
from dataclasses import dataclass, asdict

@dataclass
class ModuleChange:
    """Represents a change to a module."""
    path: str
    change_type: str  # 'created', 'modified', 'deleted', 'renamed', 'split', 'merged'
    old_path: Optional[str] = None
    related_modules: List[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.related_modules is None:
            self.related_modules = []

class RefactoringTracker:
    """Tracks module refactorings and maintains mapping."""
    
    def __init__(self, tracking_file: str = "refactoring_history.json"):
        self.tracking_file = tracking_file
        self.history = self.load_history()
        self.module_hashes = {}  # Track content hashes to detect moves
    
    def load_history(self) -> Dict:
        """Load refactoring history from file."""
        if Path(self.tracking_file).exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {"changes": [], "module_mappings": {}}
    
    def save_history(self):
        """Save refactoring history to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_module_hash(self, file_path: Path) -> str:
        """Get hash of module's significant content (functions/classes)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST and extract function/class signatures
            tree = ast.parse(content)
            signatures = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    signatures.append(f"def {node.name}")
                elif isinstance(node, ast.ClassDef):
                    signatures.append(f"class {node.name}")
            
            # Hash the signatures
            return hashlib.md5('\n'.join(sorted(signatures)).encode()).hexdigest()
        except:
            return None
    
    def detect_refactoring(self, old_modules: Set[Path], new_modules: Set[Path]) -> List[ModuleChange]:
        """Detect refactoring patterns between old and new module sets."""
        changes = []
        
        # Detect deletions and potential renames
        deleted = old_modules - new_modules
        created = new_modules - old_modules
        
        # Build hash map for similarity detection
        old_hashes = {}
        for module in deleted:
            h = self.get_module_hash(module)
            if h:
                old_hashes[h] = module
        
        new_hashes = {}
        for module in created:
            h = self.get_module_hash(module)
            if h:
                new_hashes[h] = module
        
        # Detect renames (same hash, different path)
        for h, new_path in new_hashes.items():
            if h in old_hashes:
                old_path = old_hashes[h]
                changes.append(ModuleChange(
                    path=str(new_path),
                    change_type='renamed',
                    old_path=str(old_path)
                ))
                deleted.discard(old_path)
                created.discard(new_path)
        
        # Detect splits and merges by analyzing content similarity
        if deleted and created:
            # Check for splits (one old -> multiple new)
            for old_module in deleted:
                similar_new = []
                old_content = self.get_module_content(old_module)
                
                for new_module in created:
                    new_content = self.get_module_content(new_module)
                    similarity = self.calculate_similarity(old_content, new_content)
                    if similarity > 0.3:  # 30% similarity threshold
                        similar_new.append(str(new_module))
                
                if len(similar_new) > 1:
                    changes.append(ModuleChange(
                        path=str(old_module),
                        change_type='split',
                        related_modules=similar_new
                    ))
                    for new in similar_new:
                        created.discard(Path(new))
            
            # Check for merges (multiple old -> one new)
            for new_module in list(created):
                similar_old = []
                new_content = self.get_module_content(new_module)
                
                for old_module in deleted:
                    old_content = self.get_module_content(old_module)
                    similarity = self.calculate_similarity(old_content, new_content)
                    if similarity > 0.3:
                        similar_old.append(str(old_module))
                
                if len(similar_old) > 1:
                    changes.append(ModuleChange(
                        path=str(new_module),
                        change_type='merged',
                        related_modules=similar_old
                    ))
                    created.discard(new_module)
        
        # Remaining are simple creates/deletes
        for module in created:
            changes.append(ModuleChange(path=str(module), change_type='created'))
        for module in deleted:
            changes.append(ModuleChange(path=str(module), change_type='deleted'))
        
        return changes
    
    def get_module_content(self, path: Path) -> str:
        """Get module content for comparison."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ""
    
    def calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity ratio between two contents."""
        return difflib.SequenceMatcher(None, content1, content2).ratio()
    
    def track_changes(self, changes: List[ModuleChange]):
        """Track detected changes in history."""
        for change in changes:
            self.history["changes"].append(asdict(change))
            
            # Update mappings
            if change.change_type == 'renamed':
                self.history["module_mappings"][change.old_path] = change.path
            elif change.change_type == 'split':
                for new_path in change.related_modules:
                    self.history["module_mappings"][change.path] = new_path
            elif change.change_type == 'merged':
                for old_path in change.related_modules:
                    self.history["module_mappings"][old_path] = change.path
        
        self.save_history()

class TestMaintainer:
    """Maintains tests in response to code changes."""
    
    def __init__(self, refactoring_tracker: RefactoringTracker):
        self.tracker = refactoring_tracker
        self.test_dir = Path("tests/unit")
        self.module_dir = Path("multi_coder_analysis")
    
    def update_test_imports(self, change: ModuleChange):
        """Update test imports after refactoring."""
        if change.change_type == 'renamed':
            # Find test file
            old_module_name = Path(change.old_path).stem
            test_file = self.test_dir / f"test_{old_module_name}_intelligent.py"
            
            if test_file.exists():
                # Read test
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update imports
                old_import = f"from {change.old_path.replace('/', '.').replace('\\', '.').replace('.py', '')}"
                new_import = f"from {change.path.replace('/', '.').replace('\\', '.').replace('.py', '')}"
                content = content.replace(old_import, new_import)
                
                # Rename test file
                new_module_name = Path(change.path).stem
                new_test_file = self.test_dir / f"test_{new_module_name}_intelligent.py"
                
                # Write updated test
                with open(new_test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Remove old test
                test_file.unlink()
                
                print(f"[UPDATE] Test imports updated for {old_module_name} -> {new_module_name}")
    
    def generate_test_for_new_module(self, module_path: str):
        """Generate test for newly created module."""
        # This would call your test generation system
        print(f"[GENERATE] Creating test for new module: {module_path}")
        # subprocess.run([sys.executable, "enhanced_self_healing_verifier.py", "--module", module_path])
    
    def handle_split(self, change: ModuleChange):
        """Handle module split - create tests for new modules."""
        print(f"[SPLIT] Module {change.path} split into {len(change.related_modules)} modules")
        for new_module in change.related_modules:
            self.generate_test_for_new_module(new_module)
    
    def handle_merge(self, change: ModuleChange):
        """Handle module merge - combine tests."""
        print(f"[MERGE] Modules {change.related_modules} merged into {change.path}")
        # Logic to combine tests from old modules into new test

class AgenticTestMonitor:
    """Main monitoring system that runs continuously or on schedule."""
    
    def __init__(self, 
                 check_interval_minutes: int = 120,
                 after_idle_minutes: int = 10):
        self.check_interval = check_interval_minutes
        self.after_idle = after_idle_minutes
        self.tracker = RefactoringTracker()
        self.maintainer = TestMaintainer(self.tracker)
        self.last_check = datetime.now()
        self.module_snapshot = self.get_current_modules()
    
    def get_current_modules(self) -> Set[Path]:
        """Get current set of Python modules."""
        base_dir = Path("multi_coder_analysis")
        return set(base_dir.rglob("*.py")) if base_dir.exists() else set()
    
    def get_last_git_commit_time(self) -> datetime:
        """Get time of last git commit."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%at"],
                capture_output=True,
                text=True
            )
            timestamp = int(result.stdout.strip())
            return datetime.fromtimestamp(timestamp)
        except:
            return datetime.now()
    
    def check_for_changes(self):
        """Check for code changes and maintain tests."""
        print(f"\n[MONITOR] Checking for changes at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get current modules
        current_modules = self.get_current_modules()
        
        # Detect changes
        changes = self.tracker.detect_refactoring(self.module_snapshot, current_modules)
        
        if changes:
            print(f"[MONITOR] Detected {len(changes)} changes")
            
            # Track changes
            self.tracker.track_changes(changes)
            
            # Handle each change
            for change in changes:
                if change.change_type == 'renamed':
                    self.maintainer.update_test_imports(change)
                elif change.change_type == 'created':
                    self.maintainer.generate_test_for_new_module(change.path)
                elif change.change_type == 'split':
                    self.maintainer.handle_split(change)
                elif change.change_type == 'merged':
                    self.maintainer.handle_merge(change)
                elif change.change_type == 'modified':
                    # Could trigger test verification here
                    print(f"[MONITOR] Module modified: {change.path}")
        else:
            print("[MONITOR] No changes detected")
        
        # Update snapshot
        self.module_snapshot = current_modules
        self.last_check = datetime.now()
    
    def run_after_idle(self):
        """Run after detecting idle period."""
        last_commit = self.get_last_git_commit_time()
        idle_time = datetime.now() - last_commit
        
        if idle_time > timedelta(minutes=self.after_idle):
            print(f"[MONITOR] Idle for {idle_time.total_seconds()/60:.0f} minutes, running maintenance")
            self.check_for_changes()
            return True
        return False
    
    def run_continuous(self):
        """Run continuously with periodic checks."""
        print(f"[MONITOR] Starting continuous monitoring (check every {self.check_interval} minutes)")
        print("[MONITOR] Press Ctrl+C to stop")
        
        # Schedule periodic checks
        schedule.every(self.check_interval).minutes.do(self.check_for_changes)
        
        # Also check after idle periods
        schedule.every(5).minutes.do(self.run_after_idle)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n[MONITOR] Stopped")
    
    def run_once(self):
        """Run a single check."""
        self.check_for_changes()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic Test Monitor")
    parser.add_argument("--mode", choices=["once", "continuous", "after-idle"], 
                       default="once", help="Running mode")
    parser.add_argument("--interval", type=int, default=120,
                       help="Check interval in minutes (continuous mode)")
    parser.add_argument("--idle", type=int, default=10,
                       help="Minutes of idle before running (after-idle mode)")
    
    args = parser.parse_args()
    
    monitor = AgenticTestMonitor(
        check_interval_minutes=args.interval,
        after_idle_minutes=args.idle
    )
    
    if args.mode == "once":
        monitor.run_once()
    elif args.mode == "continuous":
        monitor.run_continuous()
    elif args.mode == "after-idle":
        if monitor.run_after_idle():
            print("[MONITOR] Maintenance complete")
        else:
            print("[MONITOR] Not idle long enough")

if __name__ == "__main__":
    main()