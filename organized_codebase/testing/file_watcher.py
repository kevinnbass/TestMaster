"""
File Watcher with Agency-Swarm Callback Patterns

Inspired by Agency-Swarm's SettingsCallbacks system for real-time
file monitoring and event handling.

Features:
- Watchdog-based file system monitoring
- Callback-based event handling
- Git integration for commit tracking
- Filter support for relevant files only
"""

import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import queue
import subprocess

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ Watchdog not available. Install with: pip install watchdog")

from core.layer_manager import requires_layer
from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state
from core.tracking_manager import get_tracking_manager, track_operation


class ChangeType(Enum):
    """Types of file changes."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChangeEvent:
    """File change event information."""
    file_path: str
    change_type: ChangeType
    timestamp: datetime
    is_directory: bool = False
    old_path: Optional[str] = None  # For move events
    file_size: Optional[int] = None
    git_info: Optional[Dict[str, Any]] = None


@dataclass
class WatcherCallbacks:
    """Callback functions for file events (Agency-Swarm pattern)."""
    on_file_created: Optional[Callable[[FileChangeEvent], None]] = None
    on_file_modified: Optional[Callable[[FileChangeEvent], None]] = None
    on_file_deleted: Optional[Callable[[FileChangeEvent], None]] = None
    on_file_moved: Optional[Callable[[FileChangeEvent], None]] = None
    on_test_file_changed: Optional[Callable[[FileChangeEvent], None]] = None
    on_source_file_changed: Optional[Callable[[FileChangeEvent], None]] = None
    on_git_commit: Optional[Callable[[Dict[str, Any]], None]] = None


class TestMasterFileHandler(FileSystemEventHandler):
    """Custom file system event handler for TestMaster."""
    
    def __init__(self, file_watcher: 'FileWatcher'):
        super().__init__()
        self.file_watcher = file_watcher
        self.last_event_time = {}
        self.debounce_seconds = 0.5  # Debounce rapid events
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if self._should_process_event(event):
            change_event = self._create_change_event(event, ChangeType.CREATED)
            self.file_watcher._handle_event(change_event)
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if self._should_process_event(event) and not event.is_directory:
            # Debounce rapid modifications
            now = time.time()
            last_time = self.last_event_time.get(event.src_path, 0)
            
            if now - last_time > self.debounce_seconds:
                self.last_event_time[event.src_path] = now
                change_event = self._create_change_event(event, ChangeType.MODIFIED)
                self.file_watcher._handle_event(change_event)
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events."""
        if self._should_process_event(event):
            change_event = self._create_change_event(event, ChangeType.DELETED)
            self.file_watcher._handle_event(change_event)
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move events."""
        if self._should_process_event(event):
            change_event = FileChangeEvent(
                file_path=event.dest_path,
                change_type=ChangeType.MOVED,
                timestamp=datetime.now(),
                is_directory=event.is_directory,
                old_path=event.src_path
            )
            self.file_watcher._handle_event(change_event)
    
    def _should_process_event(self, event: FileSystemEvent) -> bool:
        """Check if event should be processed."""
        path = Path(event.src_path)
        
        # Skip hidden files and directories
        if any(part.startswith('.') for part in path.parts):
            if not path.name.endswith('.py'):  # Allow .py files in .hidden dirs
                return False
        
        # Skip common ignore patterns
        ignore_patterns = {
            '__pycache__', '.git', '.vscode', '.idea', 'node_modules',
            '.pytest_cache', '.coverage', '.tox', 'venv', '.env'
        }
        
        if any(pattern in str(path) for pattern in ignore_patterns):
            return False
        
        # Only process relevant file types
        if not event.is_directory:
            relevant_extensions = {'.py', '.yaml', '.yml', '.json', '.md', '.txt', '.cfg', '.ini'}
            if path.suffix not in relevant_extensions:
                return False
        
        return True
    
    def _create_change_event(self, event: FileSystemEvent, change_type: ChangeType) -> FileChangeEvent:
        """Create a FileChangeEvent from a watchdog event."""
        path = Path(event.src_path)
        
        # Get file size if it exists
        file_size = None
        if not event.is_directory and path.exists():
            try:
                file_size = path.stat().st_size
            except:
                pass
        
        return FileChangeEvent(
            file_path=str(path),
            change_type=change_type,
            timestamp=datetime.now(),
            is_directory=event.is_directory,
            file_size=file_size
        )


class FileWatcher:
    """
    Real-time file system watcher with callback support.
    
    Uses Agency-Swarm's callback pattern for flexible event handling
    and integrates with git for commit tracking.
    """
    
    @requires_layer("layer2_monitoring", "file_monitoring")
    def __init__(self, watch_paths: Union[str, List[str]], callbacks: Optional[WatcherCallbacks] = None):
        """
        Initialize file watcher.
        
        Args:
            watch_paths: Directory(ies) to watch
            callbacks: Callback functions for events
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError("Watchdog library required for file monitoring")
        
        self.watch_paths = [Path(p) for p in (watch_paths if isinstance(watch_paths, list) else [watch_paths])]
        self.callbacks = callbacks or WatcherCallbacks()
        
        # Initialize watchdog
        self.observer = Observer()
        self.handler = TestMasterFileHandler(self)
        
        # Event queue for processing
        self.event_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        # Git integration
        self.git_enabled = self._check_git_available()
        self.last_commit_hash = self._get_current_commit_hash()
        
        # Enhanced features integration
        self._setup_enhanced_features()
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'files_created': 0,
            'files_modified': 0,
            'files_deleted': 0,
            'files_moved': 0,
            'start_time': None
        }
    
    def _setup_enhanced_features(self):
        """Setup enhanced monitoring features."""
        # Shared state integration
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
            print("   SharedState integration enabled")
        else:
            self.shared_state = None
        
        # Tracking manager integration
        if FeatureFlags.is_enabled('layer2_monitoring', 'tracking_manager'):
            self.tracking_manager = get_tracking_manager()
            print("   Tracking manager integration enabled")
        else:
            self.tracking_manager = None
    
    def start(self):
        """Start file monitoring."""
        if self.is_running:
            print("⚠️ File watcher is already running")
            return
        
        print("🔍 Starting file system monitoring...")
        
        # Start event processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
        
        # Schedule watchdog observers
        for watch_path in self.watch_paths:
            if watch_path.exists():
                self.observer.schedule(self.handler, str(watch_path), recursive=True)
                print(f"   📁 Watching: {watch_path}")
            else:
                print(f"⚠️ Watch path does not exist: {watch_path}")
        
        # Start watchdog observer
        self.observer.start()
        self.stats['start_time'] = datetime.now()
        
        print("✅ File watcher started successfully")
        
        # Check for git changes periodically
        if self.git_enabled:
            self._start_git_monitoring()
    
    def stop(self):
        """Stop file monitoring."""
        if not self.is_running:
            return
        
        print("🛑 Stopping file system monitoring...")
        
        # Stop watchdog observer
        self.observer.stop()
        self.observer.join()
        
        # Stop event processing
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        print("✅ File watcher stopped")
    
    @track_operation("file_watcher", "handle_event")
    def _handle_event(self, event: FileChangeEvent):
        """Handle a file change event with enhanced tracking."""
        # Track event in shared state if enabled
        if self.shared_state:
            self.shared_state.increment("file_events_handled")
            self.shared_state.append("recent_file_events", {
                "file": event.file_path,
                "type": event.change_type.value,
                "timestamp": event.timestamp.isoformat(),
                "size": getattr(event, 'file_size', None)
            })
        
        self.event_queue.put(event)
    
    def _process_events(self):
        """Process events from the queue."""
        while self.is_running:
            try:
                # Get event with timeout
                event = self.event_queue.get(timeout=1)
                self._dispatch_event(event)
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Error processing event: {e}")
    
    @track_operation("file_watcher", "dispatch_event")
    def _dispatch_event(self, event: FileChangeEvent):
        """Dispatch event to appropriate callbacks with enhanced tracking."""
        self.stats['events_processed'] += 1
        
        # Track operation start
        if self.tracking_manager:
            operation_data = {
                'file_path': event.file_path,
                'change_type': event.change_type.value,
                'is_directory': event.is_directory,
                'file_size': getattr(event, 'file_size', None)
            }
            
            self.tracking_manager.track_operation(
                run_id=f"dispatch_{int(time.time() * 1000)}",
                component="file_watcher",
                operation="event_dispatch",
                inputs=operation_data
            )
        
        # Update specific stats
        if event.change_type == ChangeType.CREATED:
            self.stats['files_created'] += 1
        elif event.change_type == ChangeType.MODIFIED:
            self.stats['files_modified'] += 1
        elif event.change_type == ChangeType.DELETED:
            self.stats['files_deleted'] += 1
        elif event.change_type == ChangeType.MOVED:
            self.stats['files_moved'] += 1
        
        # Add git information
        if self.git_enabled:
            event.git_info = self._get_git_info_for_file(event.file_path)
        
        # Call general callbacks
        callback_map = {
            ChangeType.CREATED: self.callbacks.on_file_created,
            ChangeType.MODIFIED: self.callbacks.on_file_modified,
            ChangeType.DELETED: self.callbacks.on_file_deleted,
            ChangeType.MOVED: self.callbacks.on_file_moved
        }
        
        callback = callback_map.get(event.change_type)
        if callback:
            try:
                callback(event)
            except Exception as e:
                print(f"⚠️ Error in callback for {event.change_type}: {e}")
        
        # Call specialized callbacks
        if self._is_test_file(event.file_path) and self.callbacks.on_test_file_changed:
            try:
                self.callbacks.on_test_file_changed(event)
            except Exception as e:
                print(f"⚠️ Error in test file callback: {e}")
        
        elif self._is_source_file(event.file_path) and self.callbacks.on_source_file_changed:
            try:
                self.callbacks.on_source_file_changed(event)
            except Exception as e:
                print(f"⚠️ Error in source file callback: {e}")
        
        # Log significant events
        if not event.is_directory:
            print(f"📝 {event.change_type.value.title()}: {Path(event.file_path).name}")
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file."""
        path = Path(file_path)
        name = path.name.lower()
        return name.startswith('test_') or name.endswith('_test.py') or 'test' in str(path.parent).lower()
    
    def _is_source_file(self, file_path: str) -> bool:
        """Check if file is a source file."""
        path = Path(file_path)
        return (path.suffix == '.py' and 
                not self._is_test_file(file_path) and
                not path.name.startswith('_'))
    
    def _check_git_available(self) -> bool:
        """Check if git is available."""
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except:
            return False
    
    def _get_current_commit_hash(self) -> Optional[str]:
        """Get current git commit hash."""
        if not self.git_enabled:
            return None
        
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return None
    
    def _get_git_info_for_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get git information for a specific file."""
        if not self.git_enabled:
            return None
        
        try:
            # Get last commit info for file
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%H|%an|%ad|%s', '--', file_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split('|', 3)
                if len(parts) >= 4:
                    return {
                        'last_commit_hash': parts[0],
                        'last_author': parts[1],
                        'last_date': parts[2],
                        'last_message': parts[3]
                    }
        except:
            pass
        
        return None
    
    def _start_git_monitoring(self):
        """Start periodic git monitoring for commits."""
        def monitor_git():
            while self.is_running:
                time.sleep(30)  # Check every 30 seconds
                
                if not self.is_running:
                    break
                
                current_hash = self._get_current_commit_hash()
                if current_hash and current_hash != self.last_commit_hash:
                    # New commit detected
                    commit_info = self._get_commit_info(current_hash)
                    if commit_info and self.callbacks.on_git_commit:
                        try:
                            self.callbacks.on_git_commit(commit_info)
                        except Exception as e:
                            print(f"⚠️ Error in git commit callback: {e}")
                    
                    self.last_commit_hash = current_hash
        
        git_thread = threading.Thread(target=monitor_git, daemon=True)
        git_thread.start()
    
    def _get_commit_info(self, commit_hash: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a commit."""
        try:
            result = subprocess.run(
                ['git', 'show', '--format=%H|%an|%ae|%ad|%s', '--name-only', commit_hash],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    commit_line = lines[0]
                    parts = commit_line.split('|', 4)
                    
                    if len(parts) >= 5:
                        # Find where file list starts
                        file_list_start = 1
                        for i, line in enumerate(lines[1:], 1):
                            if not line.strip():
                                file_list_start = i + 1
                                break
                        
                        changed_files = [
                            line.strip() for line in lines[file_list_start:]
                            if line.strip()
                        ]
                        
                        return {
                            'hash': parts[0],
                            'author': parts[1],
                            'email': parts[2],
                            'date': parts[3],
                            'message': parts[4],
                            'changed_files': changed_files,
                            'timestamp': datetime.now()
                        }
        except:
            pass
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file watcher statistics."""
        stats = dict(self.stats)
        
        if stats['start_time']:
            uptime = datetime.now() - stats['start_time']
            stats['uptime_seconds'] = uptime.total_seconds()
            
            # Calculate rates
            if stats['uptime_seconds'] > 0:
                stats['events_per_minute'] = (stats['events_processed'] / stats['uptime_seconds']) * 60
        
        stats['is_running'] = self.is_running
        stats['watched_paths'] = [str(p) for p in self.watch_paths]
        stats['git_enabled'] = self.git_enabled
        
        return stats
    
    def add_watch_path(self, path: Union[str, Path]):
        """Add a new path to watch."""
        path = Path(path)
        if path not in self.watch_paths and path.exists():
            self.watch_paths.append(path)
            if self.is_running:
                self.observer.schedule(self.handler, str(path), recursive=True)
                print(f"📁 Added watch path: {path}")
    
    def remove_watch_path(self, path: Union[str, Path]):
        """Remove a path from watching."""
        path = Path(path)
        if path in self.watch_paths:
            self.watch_paths.remove(path)
            # Note: watchdog doesn't easily support removing individual paths
            # Would need to restart observer to fully remove
            print(f"📁 Removed watch path: {path}")


# Convenience function for quick setup
def create_test_watcher(source_dir: str, test_dir: str, 
                       on_change: Optional[Callable[[FileChangeEvent], None]] = None) -> FileWatcher:
    """
    Create a file watcher configured for test monitoring.
    
    Args:
        source_dir: Source code directory
        test_dir: Test directory
        on_change: Optional callback for any file change
        
    Returns:
        Configured FileWatcher instance
    """
    callbacks = WatcherCallbacks()
    
    if on_change:
        callbacks.on_file_created = on_change
        callbacks.on_file_modified = on_change
        callbacks.on_file_deleted = on_change
        callbacks.on_file_moved = on_change
    
    return FileWatcher(
        watch_paths=[source_dir, test_dir],
        callbacks=callbacks
    )