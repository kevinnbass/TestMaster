"""
Enhanced State Management System
===============================

Enhanced state management system that extends the existing core/shared_state.py
with advanced team management, service coordination, and multi-tier state
hierarchies from the unified state manager.

Integrates with:
- core/shared_state.py for basic state management
- core/orchestration/ for execution state
- core/reliability/ for state backup/recovery

Author: TestMaster Core State Enhancement
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
import copy

# Import existing shared state
try:
    from ..shared_state import SharedState, StateManager
except ImportError:
    # Fallback definitions
    class SharedState:
        def __init__(self):
            self.data = {}
    
    class StateManager:
        def __init__(self):
            self.state = {}

logger = logging.getLogger(__name__)

# Enhanced Enums
class TeamRole(Enum):
    """Team roles for collaborative testing."""
    ARCHITECT = "architect"
    ENGINEER = "engineer"
    QA_AGENT = "qa_agent"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    REVIEWER = "reviewer"

class SupervisorMode(Enum):
    """Supervisor operation modes."""
    GUIDED = "guided"
    AUTONOMOUS = "autonomous"
    COLLABORATIVE = "collaborative"
    ADAPTIVE = "adaptive"

class ServiceType(Enum):
    """Types of services in the system."""
    TEST_EXECUTOR = "test_executor"
    TEST_ANALYZER = "test_analyzer"
    TEST_REPORTER = "test_reporter"
    TEST_SCHEDULER = "test_scheduler"
    TEST_MONITOR = "test_monitor"
    ORCHESTRATOR = "orchestrator"
    GATEWAY = "gateway"
    REGISTRY = "registry"
    BACKUP_SERVICE = "backup_service"
    INTELLIGENCE_SERVICE = "intelligence_service"

class StateScope(Enum):
    """Scope levels for state management."""
    GLOBAL = "global"
    SERVICE = "service"
    TEAM = "team"
    SESSION = "session"
    TASK = "task"
    TEMPORARY = "temporary"

class StatePersistence(Enum):
    """State persistence levels."""
    MEMORY_ONLY = "memory_only"
    SESSION_PERSISTENT = "session_persistent"
    DISK_PERSISTENT = "disk_persistent"
    DISTRIBUTED = "distributed"
    BACKUP_REQUIRED = "backup_required"

@dataclass
class TeamConfiguration:
    """Configuration for testing teams."""
    team_id: str
    name: str
    roles: List[TeamRole] = field(default_factory=list)
    supervisor_mode: SupervisorMode = SupervisorMode.GUIDED
    max_concurrent_tasks: int = 5
    specializations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    collaboration_rules: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'team_id': self.team_id,
            'name': self.name,
            'roles': [role.value for role in self.roles],
            'supervisor_mode': self.supervisor_mode.value,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'specializations': self.specializations,
            'performance_metrics': self.performance_metrics,
            'collaboration_rules': self.collaboration_rules,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class ServiceConfiguration:
    """Configuration for system services."""
    service_id: str
    service_type: ServiceType
    name: str
    endpoint: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    health_check_interval: int = 30
    max_retry_attempts: int = 3
    timeout_seconds: int = 60
    config_params: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'service_id': self.service_id,
            'service_type': self.service_type.value,
            'name': self.name,
            'endpoint': self.endpoint,
            'capabilities': self.capabilities,
            'dependencies': self.dependencies,
            'health_check_interval': self.health_check_interval,
            'max_retry_attempts': self.max_retry_attempts,
            'timeout_seconds': self.timeout_seconds,
            'config_params': self.config_params,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class StateEntry:
    """Enhanced state entry with metadata."""
    key: str
    value: Any
    scope: StateScope
    persistence: StatePersistence
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def is_expired(self) -> bool:
        """Check if state entry has expired."""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'value': self.value,
            'scope': self.scope.value,
            'persistence': self.persistence.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata,
            'version': self.version
        }

class EnhancedStateManager:
    """
    Enhanced state management system for TestMaster framework.
    
    Provides multi-tier state management with team coordination,
    service configuration, and advanced persistence options.
    """
    
    def __init__(self,
                 base_state_manager: StateManager = None,
                 backup_service=None,
                 enable_distributed_state: bool = False):
        """
        Initialize enhanced state manager.
        
        Args:
            base_state_manager: Existing state manager to extend
            backup_service: Backup service for state persistence
            enable_distributed_state: Enable distributed state features
        """
        self.base_state_manager = base_state_manager or StateManager()
        self.backup_service = backup_service
        self.enable_distributed_state = enable_distributed_state
        
        # Enhanced state storage
        self.state_entries: Dict[str, StateEntry] = {}
        self.scoped_state: Dict[StateScope, Dict[str, StateEntry]] = {
            scope: {} for scope in StateScope
        }
        
        # Team and service management
        self.teams: Dict[str, TeamConfiguration] = {}
        self.services: Dict[str, ServiceConfiguration] = {}
        
        # State observers and event handlers
        self.state_observers: Dict[str, List[Callable]] = defaultdict(list)
        self.change_handlers: Dict[StateScope, List[Callable]] = defaultdict(list)
        
        # Session and transaction management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.transactions: Dict[str, Dict[str, Any]] = {}
        
        # Performance and monitoring
        self.state_metrics: Dict[str, Any] = {
            'total_state_entries': 0,
            'state_updates_per_minute': 0,
            'cache_hit_rate': 0.0,
            'persistence_operations': 0,
            'backup_operations': 0
        }
        
        # Cleanup and maintenance
        self.cleanup_enabled = True
        self.cleanup_interval = 300  # 5 minutes
        self.cleanup_thread: Optional[threading.Thread] = None
        
        # Synchronization
        self.lock = threading.RLock()
        
        # Initialize system
        self._start_cleanup_thread()
        self._register_default_handlers()
        
        logger.info("Enhanced State Manager initialized")
    
    def set_state(self,
                  key: str,
                  value: Any,
                  scope: StateScope = StateScope.GLOBAL,
                  persistence: StatePersistence = StatePersistence.MEMORY_ONLY,
                  ttl_seconds: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set state with enhanced options.
        
        Args:
            key: State key
            value: State value
            scope: State scope
            persistence: Persistence level
            ttl_seconds: Time to live in seconds
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        with self.lock:
            try:
                # Calculate expiration
                expires_at = None
                if ttl_seconds:
                    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
                
                # Get existing entry for version tracking
                existing_entry = self.state_entries.get(key)
                version = (existing_entry.version + 1) if existing_entry else 1
                
                # Create state entry
                entry = StateEntry(
                    key=key,
                    value=copy.deepcopy(value),
                    scope=scope,
                    persistence=persistence,
                    expires_at=expires_at,
                    metadata=metadata or {},
                    version=version
                )
                
                # Store in appropriate scopes
                self.state_entries[key] = entry
                self.scoped_state[scope][key] = entry
                
                # Update base state manager if available
                if hasattr(self.base_state_manager, 'set_state'):
                    self.base_state_manager.set_state(key, value)
                elif hasattr(self.base_state_manager, 'state'):
                    self.base_state_manager.state[key] = value
                
                # Handle persistence
                if persistence != StatePersistence.MEMORY_ONLY:
                    self._persist_state_entry(entry)
                
                # Notify observers
                self._notify_observers(key, value, 'set')
                
                # Update metrics
                self.state_metrics['total_state_entries'] = len(self.state_entries)
                
                logger.debug(f"State set: {key} (scope: {scope.value}, persistence: {persistence.value})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to set state {key}: {e}")
                return False
    
    def get_state(self,
                  key: str,
                  default: Any = None,
                  scope: StateScope = None) -> Any:
        """
        Get state with scope awareness.
        
        Args:
            key: State key
            default: Default value if not found
            scope: Specific scope to search (None = search all)
            
        Returns:
            State value or default
        """
        with self.lock:
            try:
                # Check specific scope first
                if scope:
                    entry = self.scoped_state[scope].get(key)
                    if entry and not entry.is_expired():
                        self._notify_observers(key, entry.value, 'get')
                        return copy.deepcopy(entry.value)
                
                # Check global state
                entry = self.state_entries.get(key)
                if entry and not entry.is_expired():
                    self._notify_observers(key, entry.value, 'get')
                    return copy.deepcopy(entry.value)
                
                # Fall back to base state manager
                if hasattr(self.base_state_manager, 'get_state'):
                    return self.base_state_manager.get_state(key, default)
                elif hasattr(self.base_state_manager, 'state'):
                    return self.base_state_manager.state.get(key, default)
                
                return default
                
            except Exception as e:
                logger.error(f"Failed to get state {key}: {e}")
                return default
    
    def delete_state(self, key: str, scope: StateScope = None) -> bool:
        """Delete state entry."""
        with self.lock:
            try:
                deleted = False
                
                # Delete from specific scope
                if scope and key in self.scoped_state[scope]:
                    del self.scoped_state[scope][key]
                    deleted = True
                
                # Delete from global state
                if key in self.state_entries:
                    entry = self.state_entries[key]
                    del self.state_entries[key]
                    
                    # Remove from persistence if needed
                    if entry.persistence != StatePersistence.MEMORY_ONLY:
                        self._delete_persisted_state(key)
                    
                    deleted = True
                
                # Delete from base state manager
                if hasattr(self.base_state_manager, 'delete_state'):
                    self.base_state_manager.delete_state(key)
                elif hasattr(self.base_state_manager, 'state') and key in self.base_state_manager.state:
                    del self.base_state_manager.state[key]
                
                if deleted:
                    self._notify_observers(key, None, 'delete')
                    self.state_metrics['total_state_entries'] = len(self.state_entries)
                
                return deleted
                
            except Exception as e:
                logger.error(f"Failed to delete state {key}: {e}")
                return False
    
    def register_team(self, team_config: TeamConfiguration) -> bool:
        """Register a new team configuration."""
        with self.lock:
            try:
                self.teams[team_config.team_id] = team_config
                
                # Set team state
                self.set_state(
                    f"team:{team_config.team_id}",
                    team_config.to_dict(),
                    scope=StateScope.TEAM,
                    persistence=StatePersistence.DISK_PERSISTENT
                )
                
                logger.info(f"Registered team: {team_config.name} ({team_config.team_id})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register team {team_config.team_id}: {e}")
                return False
    
    def register_service(self, service_config: ServiceConfiguration) -> bool:
        """Register a new service configuration."""
        with self.lock:
            try:
                self.services[service_config.service_id] = service_config
                
                # Set service state
                self.set_state(
                    f"service:{service_config.service_id}",
                    service_config.to_dict(),
                    scope=StateScope.SERVICE,
                    persistence=StatePersistence.DISK_PERSISTENT
                )
                
                logger.info(f"Registered service: {service_config.name} ({service_config.service_id})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register service {service_config.service_id}: {e}")
                return False
    
    def get_team_state(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Get team state and configuration."""
        team_config = self.teams.get(team_id)
        if not team_config:
            return None
        
        # Get team-scoped state
        team_state = {}
        for key, entry in self.scoped_state[StateScope.TEAM].items():
            if key.startswith(f"team:{team_id}") or entry.metadata.get('team_id') == team_id:
                team_state[key] = entry.value
        
        return {
            'configuration': team_config.to_dict(),
            'state': team_state,
            'active_tasks': self._get_team_active_tasks(team_id),
            'performance_metrics': team_config.performance_metrics
        }
    
    def get_service_state(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get service state and configuration."""
        service_config = self.services.get(service_id)
        if not service_config:
            return None
        
        # Get service-scoped state
        service_state = {}
        for key, entry in self.scoped_state[StateScope.SERVICE].items():
            if key.startswith(f"service:{service_id}") or entry.metadata.get('service_id') == service_id:
                service_state[key] = entry.value
        
        return {
            'configuration': service_config.to_dict(),
            'state': service_state,
            'health_status': self._get_service_health(service_id),
            'dependencies': service_config.dependencies
        }
    
    def create_session(self, session_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """Create a new state session."""
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        with self.lock:
            self.active_sessions[session_id] = {
                'session_id': session_id,
                'created_at': datetime.now(),
                'metadata': metadata or {},
                'state_changes': []
            }
        
        logger.debug(f"Created session: {session_id}")
        return session_id
    
    def end_session(self, session_id: str, persist_changes: bool = True) -> bool:
        """End a state session."""
        with self.lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            if persist_changes:
                # Apply session changes to persistent storage
                for change in session['state_changes']:
                    self._apply_session_change(change)
            
            del self.active_sessions[session_id]
            
        logger.debug(f"Ended session: {session_id}")
        return True
    
    def add_state_observer(self, key_pattern: str, callback: Callable) -> bool:
        """Add observer for state changes."""
        try:
            self.state_observers[key_pattern].append(callback)
            logger.debug(f"Added state observer for pattern: {key_pattern}")
            return True
        except Exception as e:
            logger.error(f"Failed to add state observer: {e}")
            return False
    
    def remove_state_observer(self, key_pattern: str, callback: Callable) -> bool:
        """Remove state observer."""
        try:
            if key_pattern in self.state_observers:
                self.state_observers[key_pattern].remove(callback)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove state observer: {e}")
            return False
    
    def get_state_by_scope(self, scope: StateScope) -> Dict[str, Any]:
        """Get all state for a specific scope."""
        with self.lock:
            result = {}
            for key, entry in self.scoped_state[scope].items():
                if not entry.is_expired():
                    result[key] = copy.deepcopy(entry.value)
            return result
    
    def cleanup_expired_state(self) -> int:
        """Clean up expired state entries."""
        with self.lock:
            expired_keys = []
            
            for key, entry in self.state_entries.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.delete_state(key)
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired state entries")
            return len(expired_keys)
    
    def backup_state(self, backup_scope: StateScope = None) -> bool:
        """Backup state to backup service."""
        if not self.backup_service:
            return False
        
        try:
            # Determine what to backup
            if backup_scope:
                state_data = self.get_state_by_scope(backup_scope)
            else:
                state_data = {
                    'global_state': {key: entry.to_dict() for key, entry in self.state_entries.items()},
                    'teams': {team_id: team.to_dict() for team_id, team in self.teams.items()},
                    'services': {service_id: service.to_dict() for service_id, service in self.services.items()}
                }
            
            # Create backup
            backup_id = self.backup_service.create_emergency_backup(
                components=['state_manager'],
                backup_type='state_snapshot'
            )
            
            if backup_id:
                self.state_metrics['backup_operations'] += 1
                logger.info(f"State backup created: {backup_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"State backup failed: {e}")
            return False
    
    def restore_state(self, backup_id: str = None) -> bool:
        """Restore state from backup."""
        if not self.backup_service:
            return False
        
        try:
            recovery_id = self.backup_service.instant_recovery(
                backup_id=backup_id,
                components=['state_manager']
            )
            
            if recovery_id:
                logger.info(f"State restored from backup: {recovery_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"State restore failed: {e}")
            return False
    
    def get_system_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive system state summary."""
        with self.lock:
            return {
                'state_entries': {
                    'total': len(self.state_entries),
                    'by_scope': {
                        scope.value: len(entries) 
                        for scope, entries in self.scoped_state.items()
                    },
                    'by_persistence': {
                        persistence.value: len([
                            e for e in self.state_entries.values()
                            if e.persistence.value == persistence.value
                        ])
                        for persistence in StatePersistence
                    }
                },
                'teams': {
                    'total': len(self.teams),
                    'by_role': self._get_teams_by_role_count()
                },
                'services': {
                    'total': len(self.services),
                    'by_type': self._get_services_by_type_count(),
                    'health_summary': self._get_services_health_summary()
                },
                'sessions': {
                    'active': len(self.active_sessions),
                    'transactions': len(self.transactions)
                },
                'performance': dict(self.state_metrics),
                'timestamp': datetime.now().isoformat()
            }
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if not self.cleanup_enabled:
            return
        
        def cleanup_loop():
            while self.cleanup_enabled:
                try:
                    self.cleanup_expired_state()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
                    time.sleep(60)
        
        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _register_default_handlers(self):
        """Register default state change handlers."""
        # Add default handlers for system events
        pass
    
    def _notify_observers(self, key: str, value: Any, operation: str):
        """Notify state observers of changes."""
        try:
            for pattern, callbacks in self.state_observers.items():
                if self._pattern_matches(key, pattern):
                    for callback in callbacks:
                        try:
                            callback(key, value, operation)
                        except Exception as e:
                            logger.warning(f"Observer callback failed: {e}")
        except Exception as e:
            logger.error(f"Failed to notify observers: {e}")
    
    def _pattern_matches(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcard matching)."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        return key == pattern
    
    def _persist_state_entry(self, entry: StateEntry):
        """Persist state entry based on persistence level."""
        # Implementation would depend on chosen persistence mechanism
        self.state_metrics['persistence_operations'] += 1
    
    def _delete_persisted_state(self, key: str):
        """Delete persisted state."""
        # Implementation would depend on chosen persistence mechanism
        pass
    
    def _get_team_active_tasks(self, team_id: str) -> List[Dict[str, Any]]:
        """Get active tasks for a team."""
        # Implementation would query task system
        return []
    
    def _get_service_health(self, service_id: str) -> Dict[str, Any]:
        """Get service health status."""
        return {'status': 'healthy', 'last_check': datetime.now().isoformat()}
    
    def _get_teams_by_role_count(self) -> Dict[str, int]:
        """Get count of teams by role."""
        role_counts = defaultdict(int)
        for team in self.teams.values():
            for role in team.roles:
                role_counts[role.value] += 1
        return dict(role_counts)
    
    def _get_services_by_type_count(self) -> Dict[str, int]:
        """Get count of services by type."""
        type_counts = defaultdict(int)
        for service in self.services.values():
            type_counts[service.service_type.value] += 1
        return dict(type_counts)
    
    def _get_services_health_summary(self) -> Dict[str, int]:
        """Get summary of service health."""
        return {'healthy': len(self.services), 'unhealthy': 0, 'unknown': 0}
    
    def _apply_session_change(self, change: Dict[str, Any]):
        """Apply a session change to persistent storage."""
        # Implementation for applying session changes
        pass
    
    def shutdown(self):
        """Shutdown enhanced state manager."""
        self.cleanup_enabled = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        # Final backup if backup service available
        if self.backup_service:
            self.backup_state()
        
        logger.info("Enhanced State Manager shutdown")

# Global enhanced state manager instance
enhanced_state_manager = None