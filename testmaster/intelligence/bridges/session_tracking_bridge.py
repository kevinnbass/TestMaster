"""
Session Tracking Bridge - Agent 13

This bridge implements comprehensive session management and state persistence
for the TestMaster hybrid intelligence system, providing seamless session
tracking, state restoration, and resume capability across all components.

Key Features:
- Comprehensive session lifecycle management
- Persistent state storage with versioning and compression
- Intelligent session restoration and resume capability
- Cross-system session synchronization
- Session analytics and performance tracking
- Consensus-driven session state decisions
- Distributed session management for scalability
"""

import json
import pickle
import sqlite3
import threading
import time
import gzip
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Union, Set
from enum import Enum
from collections import defaultdict
import uuid
import hashlib
import os
from pathlib import Path

from ..consensus import AgentCoordinator, AgentVote
from ..consensus.agent_coordination import AgentRole
from ...core.shared_state import SharedState
from ...core.feature_flags import FeatureFlags


class SessionStatus(Enum):
    """Session status types."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class SessionType(Enum):
    """Session type classifications."""
    USER_INTERACTIVE = "user_interactive"      # User-driven sessions
    WORKFLOW_EXECUTION = "workflow_execution"  # Workflow sessions
    AGENT_COORDINATION = "agent_coordination"  # Agent collaboration sessions
    BACKGROUND_TASK = "background_task"        # Background processing sessions
    INTEGRATION_TEST = "integration_test"      # Integration testing sessions
    MONITORING = "monitoring"                  # Monitoring sessions
    OPTIMIZATION = "optimization"              # Optimization sessions


class StateScope(Enum):
    """State persistence scopes."""
    SESSION_LOCAL = "session_local"           # Session-specific state
    COMPONENT_SHARED = "component_shared"     # Shared across components
    GLOBAL_PERSISTENT = "global_persistent"  # Globally persistent state
    CHECKPOINT = "checkpoint"                 # Checkpoint state
    RECOVERY = "recovery"                     # Recovery state


@dataclass
class SessionMetadata:
    """Session metadata structure."""
    session_id: str
    session_type: SessionType
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10 scale
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    max_duration: Optional[timedelta] = None
    description: str = ""


@dataclass
class SessionState:
    """Session state container."""
    session_id: str
    component_id: str
    scope: StateScope
    state_key: str
    state_data: Any
    version: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None
    compressed: bool = False
    encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionCheckpoint:
    """Session checkpoint for recovery."""
    checkpoint_id: str
    session_id: str
    checkpoint_name: str
    checkpoint_data: Dict[str, Any]
    components_state: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    restoration_priority: int = 5
    recovery_instructions: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class SessionAnalytics:
    """Session performance analytics."""
    session_id: str
    duration_seconds: float
    state_operations: int
    checkpoints_created: int
    resume_attempts: int
    successful_resumes: int
    components_involved: Set[str]
    peak_memory_usage: float
    total_data_size: int
    error_count: int
    warning_count: int
    performance_score: float = 0.0


class SessionManager:
    """Core session management functionality."""
    
    def __init__(self, storage_path: str = "testmaster_sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.active_sessions: Dict[str, SessionMetadata] = {}
        self.session_states: Dict[str, Dict[str, SessionState]] = defaultdict(dict)
        self.session_checkpoints: Dict[str, List[SessionCheckpoint]] = defaultdict(list)
        self.session_analytics: Dict[str, SessionAnalytics] = {}
        
        self.lock = threading.RLock()
        
        # Performance tracking
        self.sessions_created = 0
        self.sessions_restored = 0
        self.state_operations = 0
        self.checkpoint_operations = 0
        
        self._initialize_storage()
        
        print("Session Manager initialized")
        print(f"   Storage path: {self.storage_path}")
    
    def _initialize_storage(self):
        """Initialize session storage."""
        # Create storage directories
        (self.storage_path / "sessions").mkdir(exist_ok=True)
        (self.storage_path / "states").mkdir(exist_ok=True)
        (self.storage_path / "checkpoints").mkdir(exist_ok=True)
        (self.storage_path / "archives").mkdir(exist_ok=True)
    
    def create_session(
        self,
        session_type: SessionType,
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_duration: Optional[timedelta] = None,
        description: str = ""
    ) -> str:
        """Create new session."""
        session_id = f"{session_type.value}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        metadata = SessionMetadata(
            session_id=session_id,
            session_type=session_type,
            user_id=user_id,
            workflow_id=workflow_id,
            parent_session_id=parent_session_id,
            tags=tags or [],
            max_duration=max_duration,
            description=description,
            expires_at=datetime.now() + max_duration if max_duration else None
        )
        
        with self.lock:
            self.active_sessions[session_id] = metadata
            self.sessions_created += 1
            
            # Initialize analytics
            self.session_analytics[session_id] = SessionAnalytics(
                session_id=session_id,
                duration_seconds=0.0,
                state_operations=0,
                checkpoints_created=0,
                resume_attempts=0,
                successful_resumes=0,
                components_involved=set(),
                peak_memory_usage=0.0,
                total_data_size=0,
                error_count=0,
                warning_count=0
            )
        
        # Persist session metadata
        self._persist_session_metadata(metadata)
        
        print(f"Session created: {session_id} ({session_type.value})")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session metadata."""
        with self.lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.last_accessed = datetime.now()
                return session
        
        # Try loading from storage
        return self._load_session_metadata(session_id)
    
    def update_session_status(self, session_id: str, status: SessionStatus):
        """Update session status."""
        session = self.get_session(session_id)
        if session:
            # Note: SessionMetadata doesn't have status field in current design
            # This would be tracked in analytics or separate status tracking
            analytics = self.session_analytics.get(session_id)
            if analytics:
                # Track status changes in metadata
                if 'status_history' not in analytics.__dict__:
                    analytics.__dict__['status_history'] = []
                analytics.__dict__['status_history'].append({
                    'status': status.value,
                    'timestamp': datetime.now().isoformat()
                })
    
    def set_session_state(
        self,
        session_id: str,
        component_id: str,
        state_key: str,
        state_data: Any,
        scope: StateScope = StateScope.SESSION_LOCAL,
        compressed: bool = True
    ) -> bool:
        """Set session state for component."""
        if not self.get_session(session_id):
            return False
        
        # Calculate checksum
        state_json = json.dumps(state_data, default=str, sort_keys=True)
        checksum = hashlib.md5(state_json.encode()).hexdigest()
        
        # Compress if requested
        if compressed and len(state_json) > 1024:  # Only compress larger data
            state_data = gzip.compress(state_json.encode())
            compressed = True
        else:
            compressed = False
        
        state = SessionState(
            session_id=session_id,
            component_id=component_id,
            scope=scope,
            state_key=state_key,
            state_data=state_data,
            checksum=checksum,
            compressed=compressed
        )
        
        with self.lock:
            state_dict_key = f"{component_id}_{state_key}"
            if session_id not in self.session_states:
                self.session_states[session_id] = {}
            
            # Version management
            existing_state = self.session_states[session_id].get(state_dict_key)
            if existing_state:
                state.version = existing_state.version + 1
            
            self.session_states[session_id][state_dict_key] = state
            self.state_operations += 1
            
            # Update analytics
            analytics = self.session_analytics.get(session_id)
            if analytics:
                analytics.state_operations += 1
                analytics.components_involved.add(component_id)
                if isinstance(state_data, (str, bytes)):
                    analytics.total_data_size += len(state_data)
        
        # Persist state
        self._persist_session_state(state)
        
        return True
    
    def get_session_state(
        self,
        session_id: str,
        component_id: str,
        state_key: str
    ) -> Optional[Any]:
        """Get session state for component."""
        with self.lock:
            state_dict_key = f"{component_id}_{state_key}"
            session_states = self.session_states.get(session_id, {})
            state = session_states.get(state_dict_key)
            
            if state:
                # Decompress if needed
                if state.compressed:
                    try:
                        decompressed = gzip.decompress(state.state_data)
                        return json.loads(decompressed.decode())
                    except Exception as e:
                        print(f"State decompression error: {e}")
                        return None
                else:
                    return state.state_data
        
        # Try loading from storage
        return self._load_session_state(session_id, component_id, state_key)
    
    def create_checkpoint(
        self,
        session_id: str,
        checkpoint_name: str,
        components_to_include: Optional[List[str]] = None,
        recovery_instructions: Optional[str] = None
    ) -> str:
        """Create session checkpoint."""
        if not self.get_session(session_id):
            return ""
        
        checkpoint_id = f"checkpoint_{session_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        
        # Collect component states
        components_state = {}
        with self.lock:
            session_states = self.session_states.get(session_id, {})
            
            for state_key, state in session_states.items():
                component_id = state.component_id
                
                # Filter components if specified
                if components_to_include and component_id not in components_to_include:
                    continue
                
                if component_id not in components_state:
                    components_state[component_id] = {}
                
                components_state[component_id][state.state_key] = {
                    'data': state.state_data,
                    'version': state.version,
                    'timestamp': state.timestamp.isoformat(),
                    'checksum': state.checksum,
                    'compressed': state.compressed
                }
        
        checkpoint = SessionCheckpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            checkpoint_name=checkpoint_name,
            checkpoint_data={
                'session_metadata': self.get_session(session_id).__dict__,
                'creation_context': {
                    'components_count': len(components_state),
                    'total_states': sum(len(states) for states in components_state.values()),
                    'created_by': 'session_manager'
                }
            },
            components_state=components_state,
            recovery_instructions=recovery_instructions
        )
        
        with self.lock:
            self.session_checkpoints[session_id].append(checkpoint)
            self.checkpoint_operations += 1
            
            # Update analytics
            analytics = self.session_analytics.get(session_id)
            if analytics:
                analytics.checkpoints_created += 1
        
        # Persist checkpoint
        self._persist_checkpoint(checkpoint)
        
        print(f"Checkpoint created: {checkpoint_id} for session {session_id}")
        return checkpoint_id
    
    def restore_from_checkpoint(
        self,
        session_id: str,
        checkpoint_id: Optional[str] = None,
        restore_components: Optional[List[str]] = None
    ) -> bool:
        """Restore session from checkpoint."""
        # Find checkpoint
        checkpoint = None
        if checkpoint_id:
            # Find specific checkpoint
            for cp in self.session_checkpoints.get(session_id, []):
                if cp.checkpoint_id == checkpoint_id:
                    checkpoint = cp
                    break
        else:
            # Use latest checkpoint
            checkpoints = self.session_checkpoints.get(session_id, [])
            if checkpoints:
                checkpoint = max(checkpoints, key=lambda c: c.created_at)
        
        if not checkpoint:
            # Try loading from storage
            checkpoint = self._load_latest_checkpoint(session_id)
            
        if not checkpoint:
            print(f"No checkpoint found for session {session_id}")
            return False
        
        try:
            # Restore session metadata
            metadata_dict = checkpoint.checkpoint_data.get('session_metadata', {})
            if 'created_at' in metadata_dict:
                metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
            if 'last_accessed' in metadata_dict:
                metadata_dict['last_accessed'] = datetime.fromisoformat(metadata_dict['last_accessed'])
            if 'expires_at' in metadata_dict and metadata_dict['expires_at']:
                metadata_dict['expires_at'] = datetime.fromisoformat(metadata_dict['expires_at'])
            
            # Restore component states
            restored_count = 0
            with self.lock:
                for component_id, component_states in checkpoint.components_state.items():
                    # Filter components if specified
                    if restore_components and component_id not in restore_components:
                        continue
                    
                    for state_key, state_data in component_states.items():
                        state = SessionState(
                            session_id=session_id,
                            component_id=component_id,
                            scope=StateScope.RECOVERY,
                            state_key=state_key,
                            state_data=state_data['data'],
                            version=state_data['version'],
                            timestamp=datetime.fromisoformat(state_data['timestamp']),
                            checksum=state_data['checksum'],
                            compressed=state_data['compressed']
                        )
                        
                        state_dict_key = f"{component_id}_{state_key}"
                        if session_id not in self.session_states:
                            self.session_states[session_id] = {}
                        
                        self.session_states[session_id][state_dict_key] = state
                        restored_count += 1
                
                # Update analytics
                analytics = self.session_analytics.get(session_id)
                if analytics:
                    analytics.resume_attempts += 1
                    analytics.successful_resumes += 1
                
                self.sessions_restored += 1
            
            print(f"Session {session_id} restored from checkpoint {checkpoint.checkpoint_id}")
            print(f"   Restored {restored_count} state entries")
            return True
            
        except Exception as e:
            print(f"Session restoration error: {e}")
            
            # Update analytics for failed restore
            with self.lock:
                analytics = self.session_analytics.get(session_id)
                if analytics:
                    analytics.resume_attempts += 1
                    analytics.error_count += 1
            
            return False
    
    def close_session(self, session_id: str, archive: bool = True) -> bool:
        """Close and optionally archive session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Calculate final analytics
        with self.lock:
            analytics = self.session_analytics.get(session_id)
            if analytics:
                analytics.duration_seconds = (datetime.now() - session.created_at).total_seconds()
                analytics.performance_score = self._calculate_performance_score(analytics)
        
        if archive:
            # Archive session data
            self._archive_session(session_id)
        
        # Remove from active sessions
        with self.lock:
            self.active_sessions.pop(session_id, None)
            self.session_states.pop(session_id, None)
            self.session_checkpoints.pop(session_id, None)
        
        print(f"Session closed: {session_id}")
        return True
    
    def _calculate_performance_score(self, analytics: SessionAnalytics) -> float:
        """Calculate session performance score."""
        score = 100.0
        
        # Deduct for errors and warnings
        score -= (analytics.error_count * 10)
        score -= (analytics.warning_count * 5)
        
        # Bonus for successful resumes
        if analytics.resume_attempts > 0:
            resume_success_rate = analytics.successful_resumes / analytics.resume_attempts
            score += (resume_success_rate * 10)
        
        # Bonus for checkpoints
        if analytics.checkpoints_created > 0:
            score += min(analytics.checkpoints_created * 2, 10)
        
        return max(0.0, min(100.0, score))
    
    def _persist_session_metadata(self, metadata: SessionMetadata):
        """Persist session metadata to storage."""
        try:
            metadata_path = self.storage_path / "sessions" / f"{metadata.session_id}.json"
            with open(metadata_path, 'w') as f:
                # Convert to dict and handle datetime serialization
                metadata_dict = asdict(metadata)
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                metadata_dict['last_accessed'] = metadata.last_accessed.isoformat()
                if metadata.expires_at:
                    metadata_dict['expires_at'] = metadata.expires_at.isoformat()
                if metadata.max_duration:
                    metadata_dict['max_duration_seconds'] = metadata.max_duration.total_seconds()
                metadata_dict['session_type'] = metadata.session_type.value
                
                json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            print(f"Session metadata persistence error: {e}")
    
    def _load_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Load session metadata from storage."""
        try:
            metadata_path = self.storage_path / "sessions" / f"{session_id}.json"
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Convert back to SessionMetadata
            metadata_dict['session_type'] = SessionType(metadata_dict['session_type'])
            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
            metadata_dict['last_accessed'] = datetime.fromisoformat(metadata_dict['last_accessed'])
            if metadata_dict.get('expires_at'):
                metadata_dict['expires_at'] = datetime.fromisoformat(metadata_dict['expires_at'])
            if metadata_dict.get('max_duration_seconds'):
                metadata_dict['max_duration'] = timedelta(seconds=metadata_dict.pop('max_duration_seconds'))
            
            return SessionMetadata(**metadata_dict)
            
        except Exception as e:
            print(f"Session metadata loading error: {e}")
            return None
    
    def _persist_session_state(self, state: SessionState):
        """Persist session state to storage."""
        try:
            state_dir = self.storage_path / "states" / state.session_id
            state_dir.mkdir(exist_ok=True)
            
            state_path = state_dir / f"{state.component_id}_{state.state_key}_{state.version}.pkl"
            
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
                
        except Exception as e:
            print(f"Session state persistence error: {e}")
    
    def _load_session_state(self, session_id: str, component_id: str, state_key: str) -> Optional[Any]:
        """Load session state from storage."""
        try:
            state_dir = self.storage_path / "states" / session_id
            if not state_dir.exists():
                return None
            
            # Find latest version
            pattern = f"{component_id}_{state_key}_*.pkl"
            state_files = list(state_dir.glob(pattern))
            
            if not state_files:
                return None
            
            # Get latest version
            latest_file = max(state_files, key=lambda f: int(f.stem.split('_')[-1]))
            
            with open(latest_file, 'rb') as f:
                state = pickle.load(f)
                
                # Decompress if needed
                if state.compressed:
                    try:
                        decompressed = gzip.decompress(state.state_data)
                        return json.loads(decompressed.decode())
                    except Exception as e:
                        print(f"State decompression error: {e}")
                        return None
                else:
                    return state.state_data
                    
        except Exception as e:
            print(f"Session state loading error: {e}")
            return None
    
    def _persist_checkpoint(self, checkpoint: SessionCheckpoint):
        """Persist checkpoint to storage."""
        try:
            checkpoint_dir = self.storage_path / "checkpoints" / checkpoint.session_id
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"{checkpoint.checkpoint_id}.pkl"
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
                
        except Exception as e:
            print(f"Checkpoint persistence error: {e}")
    
    def _load_latest_checkpoint(self, session_id: str) -> Optional[SessionCheckpoint]:
        """Load latest checkpoint from storage."""
        try:
            checkpoint_dir = self.storage_path / "checkpoints" / session_id
            if not checkpoint_dir.exists():
                return None
            
            checkpoint_files = list(checkpoint_dir.glob("*.pkl"))
            if not checkpoint_files:
                return None
            
            # Get latest checkpoint
            latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            print(f"Checkpoint loading error: {e}")
            return None
    
    def _archive_session(self, session_id: str):
        """Archive session data."""
        try:
            archive_dir = self.storage_path / "archives" / session_id
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Archive session metadata
            session_file = self.storage_path / "sessions" / f"{session_id}.json"
            if session_file.exists():
                session_file.rename(archive_dir / "metadata.json")
            
            # Archive states
            states_dir = self.storage_path / "states" / session_id
            if states_dir.exists():
                import shutil
                shutil.move(str(states_dir), str(archive_dir / "states"))
            
            # Archive checkpoints
            checkpoints_dir = self.storage_path / "checkpoints" / session_id
            if checkpoints_dir.exists():
                import shutil
                shutil.move(str(checkpoints_dir), str(archive_dir / "checkpoints"))
            
            print(f"Session archived: {session_id}")
            
        except Exception as e:
            print(f"Session archiving error: {e}")
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get session manager metrics."""
        with self.lock:
            return {
                "sessions_created": self.sessions_created,
                "sessions_restored": self.sessions_restored,
                "active_sessions": len(self.active_sessions),
                "state_operations": self.state_operations,
                "checkpoint_operations": self.checkpoint_operations,
                "total_checkpoints": sum(len(cps) for cps in self.session_checkpoints.values()),
                "analytics_available": len(self.session_analytics),
                "storage_path": str(self.storage_path)
            }


class SessionTrackingBridge:
    """Main session tracking bridge orchestrator."""
    
    def __init__(self, storage_path: str = "testmaster_sessions"):
        self.enabled = FeatureFlags.is_enabled('layer4_bridges', 'session_tracking')
        
        # Core components
        self.session_manager = SessionManager(storage_path)
        self.shared_state = SharedState()
        self.coordinator = AgentCoordinator()
        
        # Bridge state
        self.component_sessions: Dict[str, Set[str]] = defaultdict(set)
        self.session_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.auto_checkpoint_intervals: Dict[str, int] = {}  # session_id -> seconds
        
        # Performance tracking
        self.start_time = datetime.now()
        self.bridge_operations = 0
        
        if not self.enabled:
            return
        
        self._setup_session_integrations()
        self._start_background_tasks()
        
        print("Session Tracking Bridge initialized")
        print(f"   Storage enabled: {self.enabled}")
        print(f"   Auto-checkpoint support: True")
    
    def _setup_session_integrations(self):
        """Setup integrations with existing TestMaster systems."""
        # Register with shared state for cross-component coordination
        self.shared_state.set("session_bridge_active", {
            "bridge_id": "session_tracking",
            "capabilities": ["session_management", "state_persistence", "resume_capability"],
            "started_at": self.start_time.isoformat()
        })
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        def background_worker():
            while self.enabled:
                try:
                    # Auto-checkpoint sessions
                    self._process_auto_checkpoints()
                    
                    # Clean up expired sessions
                    self._cleanup_expired_sessions()
                    
                    # Update session analytics
                    self._update_session_analytics()
                    
                    time.sleep(60)  # Run every minute
                    
                except Exception as e:
                    print(f"Session background task error: {e}")
                    time.sleep(30)
        
        background_thread = threading.Thread(target=background_worker, daemon=True)
        background_thread.start()
    
    def start_session(
        self,
        component_id: str,
        session_type: SessionType = SessionType.USER_INTERACTIVE,
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        max_duration: Optional[timedelta] = None,
        description: str = "",
        auto_checkpoint_interval: Optional[int] = None
    ) -> str:
        """Start new session for component."""
        session_id = self.session_manager.create_session(
            session_type=session_type,
            user_id=user_id,
            workflow_id=workflow_id,
            description=description,
            max_duration=max_duration
        )
        
        # Register component with session
        self.component_sessions[component_id].add(session_id)
        
        # Setup auto-checkpoint if requested
        if auto_checkpoint_interval:
            self.auto_checkpoint_intervals[session_id] = auto_checkpoint_interval
        
        # Notify subscribers
        self._notify_session_event("session_started", session_id, {
            "component_id": component_id,
            "session_type": session_type.value,
            "user_id": user_id,
            "workflow_id": workflow_id
        })
        
        self.bridge_operations += 1
        return session_id
    
    def save_component_state(
        self,
        session_id: str,
        component_id: str,
        state_data: Dict[str, Any],
        state_scope: StateScope = StateScope.SESSION_LOCAL
    ) -> bool:
        """Save component state to session."""
        success = True
        
        # Save each state item
        for key, value in state_data.items():
            if not self.session_manager.set_session_state(
                session_id, component_id, key, value, state_scope
            ):
                success = False
        
        if success:
            self._notify_session_event("state_saved", session_id, {
                "component_id": component_id,
                "state_keys": list(state_data.keys()),
                "scope": state_scope.value
            })
        
        self.bridge_operations += 1
        return success
    
    def load_component_state(
        self,
        session_id: str,
        component_id: str,
        state_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Load component state from session."""
        if not state_keys:
            # Try to determine state keys from session
            # This is a simplified approach - in practice, you'd track available keys
            state_keys = ["config", "progress", "cache", "context"]  # Common keys
        
        state_data = {}
        for key in state_keys:
            value = self.session_manager.get_session_state(session_id, component_id, key)
            if value is not None:
                state_data[key] = value
        
        self._notify_session_event("state_loaded", session_id, {
            "component_id": component_id,
            "loaded_keys": list(state_data.keys())
        })
        
        self.bridge_operations += 1
        return state_data
    
    def create_session_checkpoint(
        self,
        session_id: str,
        checkpoint_name: str,
        components: Optional[List[str]] = None,
        recovery_instructions: Optional[str] = None
    ) -> str:
        """Create session checkpoint."""
        checkpoint_id = self.session_manager.create_checkpoint(
            session_id, checkpoint_name, components, recovery_instructions
        )
        
        if checkpoint_id:
            self._notify_session_event("checkpoint_created", session_id, {
                "checkpoint_id": checkpoint_id,
                "checkpoint_name": checkpoint_name,
                "components": components
            })
        
        self.bridge_operations += 1
        return checkpoint_id
    
    def resume_session(
        self,
        session_id: str,
        component_id: str,
        checkpoint_id: Optional[str] = None
    ) -> bool:
        """Resume session from checkpoint."""
        # Restore from checkpoint
        success = self.session_manager.restore_from_checkpoint(
            session_id, checkpoint_id, [component_id] if component_id else None
        )
        
        if success:
            # Re-register component with session
            self.component_sessions[component_id].add(session_id)
            
            self._notify_session_event("session_resumed", session_id, {
                "component_id": component_id,
                "checkpoint_id": checkpoint_id,
                "success": True
            })
        
        self.bridge_operations += 1
        return success
    
    def end_session(
        self,
        session_id: str,
        component_id: str,
        create_final_checkpoint: bool = True,
        archive: bool = True
    ) -> bool:
        """End session for component."""
        # Create final checkpoint if requested
        if create_final_checkpoint:
            self.create_session_checkpoint(
                session_id,
                f"final_checkpoint_{component_id}",
                [component_id],
                "Final checkpoint before session end"
            )
        
        # Remove component from session
        self.component_sessions[component_id].discard(session_id)
        
        # Close session if no more components
        if not any(session_id in sessions for sessions in self.component_sessions.values()):
            success = self.session_manager.close_session(session_id, archive)
            
            self._notify_session_event("session_ended", session_id, {
                "component_id": component_id,
                "archived": archive,
                "success": success
            })
            
            return success
        
        self.bridge_operations += 1
        return True
    
    def subscribe_to_session_events(
        self,
        component_id: str,
        callback: Callable[[str, str, Dict[str, Any]], None]
    ):
        """Subscribe to session events."""
        self.session_subscribers[component_id].append(callback)
        print(f"Component {component_id} subscribed to session events")
    
    def _notify_session_event(self, event_type: str, session_id: str, event_data: Dict[str, Any]):
        """Notify subscribers of session events."""
        for component_id, callbacks in self.session_subscribers.items():
            for callback in callbacks:
                try:
                    callback(event_type, session_id, event_data)
                except Exception as e:
                    print(f"Session event callback error for {component_id}: {e}")
    
    def _process_auto_checkpoints(self):
        """Process auto-checkpoint intervals."""
        current_time = time.time()
        
        for session_id, interval_seconds in self.auto_checkpoint_intervals.items():
            session = self.session_manager.get_session(session_id)
            if not session:
                continue
            
            # Check if enough time has passed since last checkpoint
            checkpoints = self.session_manager.session_checkpoints.get(session_id, [])
            
            if checkpoints:
                last_checkpoint_time = max(cp.created_at for cp in checkpoints).timestamp()
                if current_time - last_checkpoint_time < interval_seconds:
                    continue
            else:
                # No checkpoints yet, check against session creation
                if current_time - session.created_at.timestamp() < interval_seconds:
                    continue
            
            # Create auto-checkpoint
            checkpoint_id = self.session_manager.create_checkpoint(
                session_id,
                f"auto_checkpoint_{int(current_time)}",
                recovery_instructions="Automatic checkpoint"
            )
            
            if checkpoint_id:
                print(f"Auto-checkpoint created: {checkpoint_id} for session {session_id}")
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.session_manager.active_sessions.items():
            if session.expires_at and current_time > session.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            print(f"Cleaning up expired session: {session_id}")
            self.session_manager.close_session(session_id, archive=True)
    
    def _update_session_analytics(self):
        """Update session analytics."""
        for session_id, analytics in self.session_manager.session_analytics.items():
            session = self.session_manager.get_session(session_id)
            if session:
                analytics.duration_seconds = (datetime.now() - session.created_at).total_seconds()
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session information."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None
        
        analytics = self.session_manager.session_analytics.get(session_id)
        checkpoints = self.session_manager.session_checkpoints.get(session_id, [])
        
        return {
            "session_metadata": {
                "session_id": session.session_id,
                "session_type": session.session_type.value,
                "user_id": session.user_id,
                "workflow_id": session.workflow_id,
                "created_at": session.created_at.isoformat(),
                "last_accessed": session.last_accessed.isoformat(),
                "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                "description": session.description,
                "tags": session.tags
            },
            "analytics": {
                "duration_seconds": analytics.duration_seconds if analytics else 0,
                "state_operations": analytics.state_operations if analytics else 0,
                "checkpoints_created": analytics.checkpoints_created if analytics else 0,
                "components_involved": list(analytics.components_involved) if analytics else [],
                "performance_score": analytics.performance_score if analytics else 0
            },
            "checkpoints": [
                {
                    "checkpoint_id": cp.checkpoint_id,
                    "checkpoint_name": cp.checkpoint_name,
                    "created_at": cp.created_at.isoformat(),
                    "components_count": len(cp.components_state)
                }
                for cp in checkpoints
            ],
            "active_components": [
                comp_id for comp_id, sessions in self.component_sessions.items()
                if session_id in sessions
            ]
        }
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive bridge metrics."""
        uptime = datetime.now() - self.start_time
        session_metrics = self.session_manager.get_session_metrics()
        
        return {
            "bridge_status": "active" if self.enabled else "disabled",
            "uptime_seconds": uptime.total_seconds(),
            "bridge_operations": self.bridge_operations,
            "session_manager_metrics": session_metrics,
            "active_component_sessions": {
                comp_id: len(sessions) for comp_id, sessions in self.component_sessions.items()
            },
            "auto_checkpoint_sessions": len(self.auto_checkpoint_intervals),
            "session_subscribers": len(self.session_subscribers),
            "storage_enabled": self.enabled
        }
    
    def optimize_session_storage(self):
        """Optimize session storage and cleanup."""
        # Archive old sessions
        cutoff_date = datetime.now() - timedelta(days=30)
        archived_count = 0
        
        for session_id in list(self.session_manager.active_sessions.keys()):
            session = self.session_manager.get_session(session_id)
            if session and session.last_accessed < cutoff_date:
                self.session_manager.close_session(session_id, archive=True)
                archived_count += 1
        
        # Store optimization results
        self.shared_state.set("session_storage_optimization", {
            "optimized_at": datetime.now().isoformat(),
            "sessions_archived": archived_count,
            "active_sessions": len(self.session_manager.active_sessions)
        })
        
        print(f"Session storage optimized: {archived_count} sessions archived")
    
    def shutdown(self):
        """Shutdown session tracking bridge."""
        # Create final checkpoints for all active sessions
        for session_id in list(self.session_manager.active_sessions.keys()):
            self.session_manager.create_checkpoint(
                session_id,
                "shutdown_checkpoint",
                recovery_instructions="Checkpoint created during bridge shutdown"
            )
        
        # Store final metrics
        final_metrics = self.get_comprehensive_metrics()
        self.shared_state.set("session_bridge_final_metrics", final_metrics)
        
        print("Session Tracking Bridge shutdown complete")


def get_session_tracking_bridge(storage_path: str = "testmaster_sessions") -> SessionTrackingBridge:
    """Get session tracking bridge instance."""
    return SessionTrackingBridge(storage_path)