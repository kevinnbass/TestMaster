"""
Telemetry Collector for TestMaster

Anonymous usage tracking and telemetry data collection
with privacy-first design following PraisonAI patterns.
"""

import os
import time
import uuid
import hashlib
import threading
import platform
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path
import json
import logging

from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state

logger = logging.getLogger(__name__)

@dataclass
class TelemetryEvent:
    """Individual telemetry event."""
    event_id: str
    event_type: str
    timestamp: datetime
    component: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    session_id: str = ""
    user_hash: str = ""

@dataclass
class SystemInfo:
    """System information for telemetry context."""
    platform: str
    python_version: str
    testmaster_version: str = "1.0.0"
    cpu_count: int = 0
    memory_gb: float = 0.0
    disk_gb: float = 0.0
    hostname_hash: str = ""

class TelemetryCollector:
    """
    Advanced telemetry collector for TestMaster.
    
    Collects anonymous usage data while respecting privacy:
    - No personal data collection
    - Anonymous session tracking
    - Feature usage statistics
    - Performance metrics
    - Error patterns (anonymized)
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize telemetry collector.
        
        Args:
            max_events: Maximum events to keep in memory
        """
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system')
        
        if not self.enabled:
            return
        
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.session_id = str(uuid.uuid4())
        self.user_hash = self._generate_user_hash()
        
        # Threading
        self.lock = threading.RLock()
        self.collection_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.events_collected = 0
        self.errors_encountered = 0
        self.last_flush = datetime.now()
        
        # System information
        self.system_info = self._collect_system_info()
        
        # Integration with shared state
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        # Event listeners
        self.event_listeners: List[Callable[[TelemetryEvent], None]] = []
        
        # Start background collection
        self._start_collection_thread()
        
        print("Telemetry collector initialized")
        print(f"   Session ID: {self.session_id[:8]}...")
        print(f"   Events buffer: {self.max_events}")
    
    def _generate_user_hash(self) -> str:
        """Generate anonymous user hash."""
        try:
            # Use system identifiers that don't expose personal info
            system_id = f"{platform.node()}-{platform.platform()}"
            return hashlib.sha256(system_id.encode()).hexdigest()[:16]
        except:
            return "anonymous"
    
    def _collect_system_info(self) -> SystemInfo:
        """Collect anonymous system information."""
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            return SystemInfo(
                platform=platform.platform(),
                python_version=platform.python_version(),
                cpu_count=os.cpu_count() or 0,
                memory_gb=round(memory_info.total / (1024**3), 2),
                disk_gb=round(disk_info.total / (1024**3), 2),
                hostname_hash=hashlib.sha256(platform.node().encode()).hexdigest()[:8]
            )
        except ImportError:
            # psutil not available, use basic info
            return SystemInfo(
                platform=platform.platform(),
                python_version=platform.python_version(),
                cpu_count=os.cpu_count() or 0
            )
        except Exception:
            # Fallback to minimal info
            return SystemInfo(
                platform="unknown",
                python_version=platform.python_version()
            )
    
    def record_event(self, event_type: str, component: str, operation: str,
                    metadata: Dict[str, Any] = None, duration_ms: float = None,
                    success: bool = True, error_message: str = None):
        """
        Record a telemetry event.
        
        Args:
            event_type: Type of event (e.g., 'test_generation', 'file_analysis')
            component: Component name (e.g., 'test_generator', 'file_watcher')
            operation: Operation name (e.g., 'generate_test', 'detect_change')
            metadata: Additional anonymous metadata
            duration_ms: Operation duration in milliseconds
            success: Whether operation succeeded
            error_message: Anonymized error message
        """
        if not self.enabled:
            return
        
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            metadata=self._sanitize_metadata(metadata or {}),
            duration_ms=duration_ms,
            success=success,
            error_message=self._sanitize_error(error_message) if error_message else None,
            session_id=self.session_id,
            user_hash=self.user_hash
        )
        
        with self.lock:
            self.events.append(event)
            self.events_collected += 1
            
            if not success:
                self.errors_encountered += 1
            
            # Update shared state if available
            if self.shared_state:
                self.shared_state.increment("telemetry_events_collected")
                if not success:
                    self.shared_state.increment("telemetry_errors_recorded")
            
            # Notify listeners
            for listener in self.event_listeners:
                try:
                    listener(event)
                except Exception:
                    # Don't let listener errors break telemetry
                    pass
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to remove sensitive information."""
        sanitized = {}
        
        for key, value in metadata.items():
            # Skip potentially sensitive keys
            if any(sensitive in key.lower() for sensitive in 
                   ['password', 'token', 'key', 'secret', 'auth', 'credential']):
                continue
            
            # Sanitize values
            if isinstance(value, str):
                # Limit string length and check for sensitive patterns
                if len(value) > 100:
                    value = value[:100] + "..."
                
                # Remove potential file paths or personal data
                if '/' in value or '\\' in value:
                    value = "[path]"
                elif '@' in value:
                    value = "[email]"
            
            elif isinstance(value, (int, float, bool)):
                # Numbers and booleans are safe
                pass
            
            elif isinstance(value, (list, tuple)):
                # Only include length for collections
                value = len(value)
            
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts
                value = self._sanitize_metadata(value)
            
            else:
                # Convert unknown types to string representation
                value = str(type(value).__name__)
            
            sanitized[key] = value
        
        return sanitized
    
    def _sanitize_error(self, error_message: str) -> str:
        """Sanitize error messages to remove sensitive information."""
        if not error_message:
            return ""
        
        # Remove file paths
        import re
        error_message = re.sub(r'[A-Za-z]:[\\\/][^\s]*', '[path]', error_message)
        error_message = re.sub(r'\/[^\s]*', '[path]', error_message)
        
        # Remove potential email addresses
        error_message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email]', error_message)
        
        # Limit length
        if len(error_message) > 200:
            error_message = error_message[:200] + "..."
        
        return error_message
    
    def add_event_listener(self, listener: Callable[[TelemetryEvent], None]):
        """Add an event listener for real-time telemetry processing."""
        if not self.enabled:
            return
        
        with self.lock:
            self.event_listeners.append(listener)
    
    def remove_event_listener(self, listener: Callable[[TelemetryEvent], None]):
        """Remove an event listener."""
        if not self.enabled:
            return
        
        with self.lock:
            if listener in self.event_listeners:
                self.event_listeners.remove(listener)
    
    def get_events(self, event_type: str = None, component: str = None,
                  limit: int = None) -> List[TelemetryEvent]:
        """
        Get collected events with optional filtering.
        
        Args:
            event_type: Filter by event type
            component: Filter by component
            limit: Maximum events to return
            
        Returns:
            List of matching events
        """
        if not self.enabled:
            return []
        
        with self.lock:
            events = list(self.events)
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if component:
            events = [e for e in events if e.component == component]
        
        # Apply limit
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get telemetry collection statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            # Count events by type and component
            event_types = {}
            components = {}
            
            for event in self.events:
                event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
                components[event.component] = components.get(event.component, 0) + 1
            
            return {
                "enabled": True,
                "session_id": self.session_id,
                "events_collected": self.events_collected,
                "errors_encountered": self.errors_encountered,
                "events_in_buffer": len(self.events),
                "max_buffer_size": self.max_events,
                "last_flush": self.last_flush.isoformat(),
                "system_info": asdict(self.system_info),
                "event_types": event_types,
                "components": components,
                "listeners": len(self.event_listeners)
            }
    
    def export_events(self, format: str = "json") -> str:
        """Export events for analysis or backup."""
        if not self.enabled:
            return "{}" if format == "json" else ""
        
        with self.lock:
            events_data = []
            for event in self.events:
                event_dict = asdict(event)
                event_dict['timestamp'] = event.timestamp.isoformat()
                events_data.append(event_dict)
        
        if format == "json":
            return json.dumps({
                "session_id": self.session_id,
                "export_timestamp": datetime.now().isoformat(),
                "system_info": asdict(self.system_info),
                "events": events_data
            }, indent=2)
        
        return str(events_data)
    
    def clear_events(self):
        """Clear all collected events."""
        if not self.enabled:
            return
        
        with self.lock:
            self.events.clear()
            self.last_flush = datetime.now()
    
    def _start_collection_thread(self):
        """Start background collection thread for periodic processing."""
        if not self.enabled:
            return
        
        def collection_worker():
            while not self.shutdown_event.is_set():
                try:
                    # Periodic cleanup and processing
                    if self.shutdown_event.wait(timeout=60):  # Check every minute
                        break
                    
                    # Perform periodic maintenance
                    self._periodic_maintenance()
                    
                except Exception:
                    # Handle errors silently
                    pass
        
        self.collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self.collection_thread.start()
    
    def _periodic_maintenance(self):
        """Perform periodic maintenance tasks."""
        with self.lock:
            # Update shared state
            if self.shared_state:
                self.shared_state.set("telemetry_last_maintenance", datetime.now().isoformat())
                self.shared_state.set("telemetry_buffer_size", len(self.events))
    
    def shutdown(self):
        """Shutdown telemetry collector."""
        if not self.enabled:
            return
        
        self.shutdown_event.set()
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)
        
        print(f"Telemetry collector shutdown - collected {self.events_collected} events")

# Global instance
_telemetry_collector: Optional[TelemetryCollector] = None

def get_telemetry_collector() -> TelemetryCollector:
    """Get the global telemetry collector instance."""
    global _telemetry_collector
    if _telemetry_collector is None:
        _telemetry_collector = TelemetryCollector()
    return _telemetry_collector