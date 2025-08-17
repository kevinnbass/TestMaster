"""
Idle Detection System with PraisonAI Telemetry Patterns

Inspired by PraisonAI's performance statistics tracking and
telemetry system for monitoring stagnation and inactivity.

Features:
- 2-hour idle threshold detection
- Module-level stagnation tracking
- Performance statistics collection
- Periodic cleanup and reporting
"""

import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
import json

from ..core.layer_manager import requires_layer


class IdleState(Enum):
    """Module idle states."""
    ACTIVE = "active"
    IDLE_WARNING = "idle_warning"  # 1.5-2 hours
    IDLE = "idle"  # 2+ hours
    STAGNANT = "stagnant"  # 24+ hours
    ABANDONED = "abandoned"  # 7+ days


@dataclass
class IdleModule:
    """Information about an idle module."""
    module_path: str
    last_modified: datetime
    idle_duration: timedelta
    idle_state: IdleState
    file_size: int
    line_count: int
    test_coverage: Optional[float] = None
    last_test_run: Optional[datetime] = None
    git_info: Optional[Dict[str, Any]] = None
    priority_score: float = 0.0


@dataclass
class IdleStatistics:
    """Idle detection statistics."""
    total_modules: int
    active_modules: int
    idle_warning_modules: int
    idle_modules: int
    stagnant_modules: int
    abandoned_modules: int
    avg_idle_duration_hours: float
    longest_idle_module: Optional[str] = None
    longest_idle_duration: Optional[timedelta] = None
    scan_timestamp: datetime = field(default_factory=datetime.now)


class IdleDetector:
    """
    Detect idle and stagnant code modules.
    
    Uses PraisonAI's performance monitoring patterns for
    statistics collection and telemetry tracking.
    """
    
    @requires_layer("layer2_monitoring", "idle_detection")
    def __init__(self, watch_paths: Union[str, List[str]], 
                 idle_threshold_hours: float = 2.0,
                 scan_interval_minutes: float = 30.0):
        """
        Initialize idle detector.
        
        Args:
            watch_paths: Directories to monitor for idle modules
            idle_threshold_hours: Hours after which a module is considered idle
            scan_interval_minutes: How often to scan for idle modules
        """
        self.watch_paths = [Path(p) for p in (watch_paths if isinstance(watch_paths, list) else [watch_paths])]
        self.idle_threshold = timedelta(hours=idle_threshold_hours)
        self.scan_interval = timedelta(minutes=scan_interval_minutes)
        
        # Module tracking (PraisonAI pattern)
        self._module_tracker: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'first_seen': datetime.now(),
            'last_modified': datetime.now(),
            'modification_count': 0,
            'scan_count': 0,
            'idle_alerts_sent': 0,
            'priority_score': 0.0
        })
        
        # Statistics tracking
        self._statistics_history: deque = deque(maxlen=100)
        self._scan_count = 0
        
        # Scanning control
        self._is_scanning = False
        self._scan_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_module_idle: Optional[Callable[[IdleModule], None]] = None
        self.on_module_active: Optional[Callable[[str], None]] = None
        self.on_statistics_update: Optional[Callable[[IdleStatistics], None]] = None
        
        print("Idle detector initialized")
        print(f"   Watching: {', '.join(str(p) for p in self.watch_paths)}")
        print(f"   Idle threshold: {idle_threshold_hours} hours")
        print(f"   Scan interval: {scan_interval_minutes} minutes")
    
    def start_monitoring(self):
        """Start continuous idle monitoring."""
        if self._is_scanning:
            print("WARNING: Idle detector is already running")
            return
        
        print(" Starting idle detection monitoring...")
        
        self._is_scanning = True
        self._scan_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._scan_thread.start()
        
        print(" Idle detection started")
    
    def stop_monitoring(self):
        """Stop idle monitoring."""
        if not self._is_scanning:
            return
        
        print(" Stopping idle detection...")
        
        self._is_scanning = False
        if self._scan_thread:
            self._scan_thread.join(timeout=10)
        
        print(" Idle detection stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._is_scanning:
            try:
                # Perform scan
                stats = self.scan_for_idle_modules()
                
                # Update statistics history
                self._statistics_history.append(stats)
                
                # Call statistics callback
                if self.on_statistics_update:
                    try:
                        self.on_statistics_update(stats)
                    except Exception as e:
                        print(f" Error in statistics callback: {e}")
                
                # Log periodic summary
                if self._scan_count % 10 == 0:  # Every 10 scans
                    self._log_summary(stats)
                
                # Wait for next scan
                if self._is_scanning:
                    time.sleep(self.scan_interval.total_seconds())
                    
            except Exception as e:
                print(f" Error in idle monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def scan_for_idle_modules(self) -> IdleStatistics:
        """
        Scan for idle modules and return statistics.
        
        Returns:
            Statistics about idle modules found
        """
        self._scan_count += 1
        print(f" Scanning for idle modules (scan #{self._scan_count})...")
        
        idle_modules = []
        active_modules = []
        current_time = datetime.now()
        
        # Scan all Python files in watch paths
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
                
            for py_file in watch_path.rglob("*.py"):
                if self._should_monitor_file(py_file):
                    module_info = self._analyze_module(py_file, current_time)
                    
                    if module_info.idle_state == IdleState.ACTIVE:
                        active_modules.append(str(py_file))
                    else:
                        idle_modules.append(module_info)
                        
                        # Trigger idle callback
                        if self.on_module_idle:
                            try:
                                self.on_module_idle(module_info)
                            except Exception as e:
                                print(f" Error in idle module callback: {e}")
        
        # Calculate statistics
        stats = self._calculate_statistics(idle_modules, active_modules, current_time)
        
        print(f" Scan complete: {stats.idle_modules} idle, {stats.active_modules} active")
        
        return stats
    
    def _should_monitor_file(self, file_path: Path) -> bool:
        """Check if file should be monitored for idle detection."""
        # Skip hidden files and common ignore patterns
        if any(part.startswith('.') for part in file_path.parts):
            return False
        
        # Skip common ignore patterns
        ignore_patterns = {
            '__pycache__', '.git', '.vscode', '.idea', 'venv', '.env',
            'node_modules', '.pytest_cache', '.coverage', '.tox'
        }
        
        if any(pattern in str(file_path) for pattern in ignore_patterns):
            return False
        
        # Skip test files (focus on source code)
        name = file_path.name.lower()
        if name.startswith('test_') or name.endswith('_test.py') or 'test' in str(file_path.parent).lower():
            return False
        
        # Skip very small files (likely stubs)
        try:
            if file_path.stat().st_size < 100:  # Less than 100 bytes
                return False
        except:
            return False
        
        return True
    
    def _analyze_module(self, file_path: Path, current_time: datetime) -> IdleModule:
        """Analyze a module for idle state."""
        try:
            # Get file stats
            stat = file_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime)
            file_size = stat.st_size
            
            # Count lines
            line_count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for line in f if line.strip())
            except:
                line_count = 0
            
            # Calculate idle duration
            idle_duration = current_time - last_modified
            
            # Determine idle state
            idle_state = self._calculate_idle_state(idle_duration)
            
            # Update tracker
            module_key = str(file_path)
            tracker = self._module_tracker[module_key]
            tracker['last_modified'] = last_modified
            tracker['scan_count'] += 1
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(
                idle_duration, file_size, line_count
            )
            tracker['priority_score'] = priority_score
            
            return IdleModule(
                module_path=str(file_path),
                last_modified=last_modified,
                idle_duration=idle_duration,
                idle_state=idle_state,
                file_size=file_size,
                line_count=line_count,
                priority_score=priority_score
            )
            
        except Exception as e:
            print(f" Error analyzing {file_path}: {e}")
            return IdleModule(
                module_path=str(file_path),
                last_modified=current_time,
                idle_duration=timedelta(0),
                idle_state=IdleState.ACTIVE,
                file_size=0,
                line_count=0
            )
    
    def _calculate_idle_state(self, idle_duration: timedelta) -> IdleState:
        """Calculate idle state based on duration."""
        hours = idle_duration.total_seconds() / 3600
        
        if hours < 1.5:
            return IdleState.ACTIVE
        elif hours < 2.0:
            return IdleState.IDLE_WARNING
        elif hours < 24:
            return IdleState.IDLE
        elif hours < 168:  # 7 days
            return IdleState.STAGNANT
        else:
            return IdleState.ABANDONED
    
    def _calculate_priority_score(self, idle_duration: timedelta, 
                                file_size: int, line_count: int) -> float:
        """Calculate priority score for idle module (0-100)."""
        # Base score from idle duration
        hours = idle_duration.total_seconds() / 3600
        duration_score = min(hours * 10, 60)  # Max 60 from duration
        
        # Size complexity score
        size_score = min(line_count / 10, 25)  # Max 25 from size
        
        # File importance score (larger files likely more important)
        importance_score = min(file_size / 1000, 15)  # Max 15 from importance
        
        return duration_score + size_score + importance_score
    
    def _calculate_statistics(self, idle_modules: List[IdleModule], 
                            active_modules: List[str], 
                            current_time: datetime) -> IdleStatistics:
        """Calculate comprehensive statistics."""
        # Count by state
        state_counts = defaultdict(int)
        for module in idle_modules:
            state_counts[module.idle_state] += 1
        
        total_modules = len(idle_modules) + len(active_modules)
        
        # Calculate average idle duration
        avg_idle_hours = 0.0
        if idle_modules:
            total_hours = sum(m.idle_duration.total_seconds() / 3600 for m in idle_modules)
            avg_idle_hours = total_hours / len(idle_modules)
        
        # Find longest idle module
        longest_idle_module = None
        longest_idle_duration = None
        if idle_modules:
            longest = max(idle_modules, key=lambda m: m.idle_duration)
            longest_idle_module = longest.module_path
            longest_idle_duration = longest.idle_duration
        
        return IdleStatistics(
            total_modules=total_modules,
            active_modules=len(active_modules),
            idle_warning_modules=state_counts[IdleState.IDLE_WARNING],
            idle_modules=state_counts[IdleState.IDLE],
            stagnant_modules=state_counts[IdleState.STAGNANT],
            abandoned_modules=state_counts[IdleState.ABANDONED],
            avg_idle_duration_hours=avg_idle_hours,
            longest_idle_module=longest_idle_module,
            longest_idle_duration=longest_idle_duration,
            scan_timestamp=current_time
        )
    
    def _log_summary(self, stats: IdleStatistics):
        """Log periodic summary of idle detection."""
        print(" Idle Detection Summary")
        print(f"    Total modules: {stats.total_modules}")
        print(f"    Active: {stats.active_modules}")
        print(f"    Idle warning: {stats.idle_warning_modules}")
        print(f"    Idle (2h+): {stats.idle_modules}")
        print(f"    Stagnant (24h+): {stats.stagnant_modules}")
        print(f"    Abandoned (7d+): {stats.abandoned_modules}")
        
        if stats.avg_idle_duration_hours > 0:
            print(f"    Avg idle duration: {stats.avg_idle_duration_hours:.1f} hours")
        
        if stats.longest_idle_module:
            hours = stats.longest_idle_duration.total_seconds() / 3600
            module_name = Path(stats.longest_idle_module).name
            print(f"    Longest idle: {module_name} ({hours:.1f}h)")
    
    def get_idle_modules(self, min_idle_hours: float = 2.0, 
                        limit: int = None) -> List[IdleModule]:
        """
        Get list of idle modules.
        
        Args:
            min_idle_hours: Minimum idle duration in hours
            limit: Maximum number of modules to return
            
        Returns:
            List of idle modules sorted by priority
        """
        stats = self.scan_for_idle_modules()
        
        # Filter and sort idle modules
        idle_modules = [
            module for module in self._get_all_idle_modules()
            if module.idle_duration.total_seconds() / 3600 >= min_idle_hours
        ]
        
        # Sort by priority score (highest first)
        idle_modules.sort(key=lambda m: m.priority_score, reverse=True)
        
        if limit:
            idle_modules = idle_modules[:limit]
        
        return idle_modules
    
    def _get_all_idle_modules(self) -> List[IdleModule]:
        """Get all currently idle modules."""
        current_time = datetime.now()
        idle_modules = []
        
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
                
            for py_file in watch_path.rglob("*.py"):
                if self._should_monitor_file(py_file):
                    module_info = self._analyze_module(py_file, current_time)
                    if module_info.idle_state != IdleState.ACTIVE:
                        idle_modules.append(module_info)
        
        return idle_modules
    
    def get_statistics(self) -> IdleStatistics:
        """Get current idle detection statistics."""
        return self.scan_for_idle_modules()
    
    def get_statistics_history(self, hours: int = 24) -> List[IdleStatistics]:
        """Get historical statistics."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            stats for stats in self._statistics_history
            if stats.scan_timestamp > cutoff
        ]
    
    def export_idle_report(self, output_path: str = "idle_report.json"):
        """Export idle module report to JSON."""
        stats = self.get_statistics()
        idle_modules = self.get_idle_modules(min_idle_hours=0)  # All idle modules
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": {
                "total_modules": stats.total_modules,
                "active_modules": stats.active_modules,
                "idle_warning_modules": stats.idle_warning_modules,
                "idle_modules": stats.idle_modules,
                "stagnant_modules": stats.stagnant_modules,
                "abandoned_modules": stats.abandoned_modules,
                "avg_idle_duration_hours": stats.avg_idle_duration_hours
            },
            "idle_modules": [
                {
                    "module_path": module.module_path,
                    "last_modified": module.last_modified.isoformat(),
                    "idle_duration_hours": module.idle_duration.total_seconds() / 3600,
                    "idle_state": module.idle_state.value,
                    "file_size": module.file_size,
                    "line_count": module.line_count,
                    "priority_score": module.priority_score
                }
                for module in idle_modules
            ]
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f" Idle report exported to {output_path}")
        except Exception as e:
            print(f" Error exporting report: {e}")
    
    def clear_statistics(self):
        """Clear statistics history."""
        self._statistics_history.clear()
        print(" Cleared idle detection statistics")


# Convenience function for quick idle checking
def quick_idle_scan(source_dir: str, 
                   idle_threshold_hours: float = 2.0) -> List[IdleModule]:
    """
    Quick scan for idle modules without continuous monitoring.
    
    Args:
        source_dir: Directory to scan
        idle_threshold_hours: Hours after which a module is considered idle
        
    Returns:
        List of idle modules
    """
    detector = IdleDetector(source_dir, idle_threshold_hours)
    return detector.get_idle_modules(min_idle_hours=idle_threshold_hours)