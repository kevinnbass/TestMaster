"""
TestMaster Monitoring Module - Layer 2

Real-time monitoring capabilities inspired by framework analysis:
- Agency-Swarm: Callback-based file monitoring 
- PraisonAI: Performance statistics tracking
- Agency-Swarm Gradio: Queue-based task management

Provides:
- Continuous codebase watching
- Idle detection (2-hour threshold)
- Periodic test scheduling
- Performance telemetry
"""

from .file_watcher import FileWatcher, FileChangeEvent
from .idle_detector import IdleDetector, IdleModule
from .test_scheduler import TestScheduler, ScheduledTest

__all__ = [
    "FileWatcher",
    "FileChangeEvent",
    "IdleDetector", 
    "IdleModule",
    "TestScheduler",
    "ScheduledTest"
]
