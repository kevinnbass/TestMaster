"""
Error Isolation System - Archive-Derived Robustness System
=========================================================

Advanced error isolation and containment system with intelligent
boundary management, cascade prevention, and automatic quarantine.

Author: Agent C Security Framework
Created: 2025-08-21
"""

import logging
import time
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import uuid
import weakref

logger = logging.getLogger(__name__)

class IsolationLevel(Enum):
    """Error isolation levels."""
    NONE = 0           # No isolation
    SOFT = 1           # Soft boundaries, warnings only
    MODERATE = 2       # Moderate isolation with monitoring
    STRICT = 3         # Strict isolation with blocking
    QUARANTINE = 4     # Complete quarantine
    EMERGENCY = 5      # Emergency isolation

class BoundaryType(Enum):
    """Types of isolation boundaries."""
    PROCESS = "process"         # Process-level boundaries
    THREAD = "thread"           # Thread-level boundaries
    COMPONENT = "component"     # Component-level boundaries
    SERVICE = "service"         # Service-level boundaries
    NETWORK = "network"         # Network-level boundaries
    RESOURCE = "resource"       # Resource-level boundaries

class IsolationTrigger(Enum):
    """Events that trigger isolation."""
    ERROR_THRESHOLD = "error_threshold"
    CASCADE_DETECTION = "cascade_detection"
    RESOURCE_VIOLATION = "resource_violation"
    SECURITY_BREACH = "security_breach"
    MANUAL_REQUEST = "manual_request"
    PREDICTIVE_ACTION = "predictive_action"
    DEPENDENCY_FAILURE = "dependency_failure"
    PERFORMANCE_ANOMALY = "performance_anomaly"

@dataclass
class IsolationBoundary:
    """Isolation boundary definition."""
    boundary_id: str
    boundary_type: BoundaryType
    isolation_level: IsolationLevel
    protected_components: Set[str]
    blocked_components: Set[str]
    allowed_interactions: Dict[str, List[str]]
    resource_limits: Dict[str, Any]
    monitoring_rules: List[Dict[str, Any]]
    created_at: datetime
    last_violation: Optional[datetime] = None
    violation_count: int = 0

@dataclass
class IsolationEvent:
    """Error isolation event record."""
    event_id: str
    boundary_id: str
    trigger: IsolationTrigger
    timestamp: datetime
    source_component: str
    target_components: List[str]
    isolation_action: str
    success: bool
    impact_assessment: Dict[str, Any]
    recovery_plan: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QuarantineZone:
    """Quarantine zone for problematic components."""
    zone_id: str
    quarantined_components: Set[str]
    quarantine_reason: str
    quarantine_start: datetime
    isolation_level: IsolationLevel
    monitoring_interval: int
    release_conditions: List[Dict[str, Any]]
    automatic_release: bool
    max_quarantine_duration: Optional[int] = None

class ErrorIsolationSystem:
    """
    Advanced error isolation system with intelligent boundary management.
    """
    
    def __init__(self, db_path: str = "data/error_isolation.db"):
        self.db_path = db_path
        
        # Initialize database
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
        # Isolation boundaries and zones
        self.isolation_boundaries: Dict[str, IsolationBoundary] = {}
        self.quarantine_zones: Dict[str, QuarantineZone] = {}
        self.component_boundaries: Dict[str, Set[str]] = defaultdict(set)
        
        # Monitoring and tracking
        self.isolation_events: deque = deque(maxlen=5000)
        self.boundary_violations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.component_health: Dict[str, Dict[str, Any]] = {}
        
        # Error tracking and pattern analysis
        self.error_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.cascade_detection: Dict[str, Dict[str, Any]] = {}
        self.interaction_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Configuration parameters
        self.max_isolation_depth = 5
        self.cascade_detection_window = 180  # 3 minutes
        self.boundary_violation_threshold = 5
        self.automatic_quarantine_enabled = True
        self.quarantine_duration_hours = 24
        
        # Statistics and metrics
        self.isolation_stats = {
            'boundaries_created': 0,
            'isolations_triggered': 0,
            'successful_isolations': 0,
            'failed_isolations': 0,
            'cascade_preventions': 0,
            'quarantine_actions': 0,
            'boundary_violations': 0,
            'automatic_releases': 0,
            'manual_overrides': 0,
            'containment_effectiveness': 100.0
        }
        
        # Background processing
        self.isolation_active = True
        self.boundary_monitor_thread = threading.Thread(target=self._boundary_monitoring_loop, daemon=True)
        self.cascade_detector_thread = threading.Thread(target=self._cascade_detection_loop, daemon=True)
        self.quarantine_manager_thread = threading.Thread(target=self._quarantine_management_loop, daemon=True)
        self.violation_analyzer_thread = threading.Thread(target=self._violation_analysis_loop, daemon=True)
        
        # Start background threads
        self.boundary_monitor_thread.start()
        self.cascade_detector_thread.start()
        self.quarantine_manager_thread.start()
        self.violation_analyzer_thread.start()
        
        # Thread safety
        self.isolation_lock = threading.RLock()
        
        # Weak references for cleanup
        self.component_references: Dict[str, weakref.ReferenceType] = {}
        
        logger.info("Error Isolation System initialized with intelligent boundary management")
    
    def _init_database(self):
        """Initialize error isolation database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Isolation boundaries table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS isolation_boundaries (
                        boundary_id TEXT PRIMARY KEY,
                        boundary_type TEXT NOT NULL,
                        isolation_level INTEGER NOT NULL,
                        protected_components TEXT NOT NULL,
                        blocked_components TEXT NOT NULL,
                        allowed_interactions TEXT,
                        resource_limits TEXT,
                        monitoring_rules TEXT,
                        created_at TEXT NOT NULL,
                        last_violation TEXT,
                        violation_count INTEGER DEFAULT 0
                    )
                ''')
                
                # Isolation events table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS isolation_events (
                        event_id TEXT PRIMARY KEY,
                        boundary_id TEXT NOT NULL,
                        trigger_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        source_component TEXT NOT NULL,
                        target_components TEXT,
                        isolation_action TEXT NOT NULL,
                        success INTEGER DEFAULT 0,
                        impact_assessment TEXT,
                        recovery_plan TEXT,
                        context TEXT
                    )
                ''')
                
                # Quarantine zones table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS quarantine_zones (
                        zone_id TEXT PRIMARY KEY,
                        quarantined_components TEXT NOT NULL,
                        quarantine_reason TEXT NOT NULL,
                        quarantine_start TEXT NOT NULL,
                        isolation_level INTEGER NOT NULL,
                        monitoring_interval INTEGER DEFAULT 60,
                        release_conditions TEXT,
                        automatic_release INTEGER DEFAULT 1,
                        max_quarantine_duration INTEGER
                    )
                ''')
                
                # Boundary violations table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS boundary_violations (
                        violation_id TEXT PRIMARY KEY,
                        boundary_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        violating_component TEXT NOT NULL,
                        violation_type TEXT NOT NULL,
                        severity INTEGER DEFAULT 3,
                        details TEXT,
                        resolved INTEGER DEFAULT 0,
                        resolution_action TEXT
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_events_boundary ON isolation_events(boundary_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON isolation_events(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_violations_boundary ON boundary_violations(boundary_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON boundary_violations(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error isolation database initialization failed: {e}")
            raise
    
    def create_isolation_boundary(self, boundary_type: BoundaryType, 
                                isolation_level: IsolationLevel,
                                protected_components: Set[str],
                                blocked_components: Optional[Set[str]] = None,
                                resource_limits: Optional[Dict[str, Any]] = None) -> str:
        """Create a new isolation boundary."""
        try:
            boundary_id = f"boundary_{boundary_type.value}_{uuid.uuid4().hex[:8]}"
            
            with self.isolation_lock:
                # Create boundary
                boundary = IsolationBoundary(
                    boundary_id=boundary_id,
                    boundary_type=boundary_type,
                    isolation_level=isolation_level,
                    protected_components=set(protected_components),
                    blocked_components=set(blocked_components or []),
                    allowed_interactions={},
                    resource_limits=resource_limits or {},
                    monitoring_rules=[],
                    created_at=datetime.now()
                )
                
                # Store boundary
                self.isolation_boundaries[boundary_id] = boundary
                
                # Update component boundary mappings
                for component in protected_components:
                    self.component_boundaries[component].add(boundary_id)
                
                # Initialize component health tracking
                for component in protected_components:
                    if component not in self.component_health:
                        self.component_health[component] = {
                            'error_count': 0,
                            'last_error': None,
                            'isolation_level': IsolationLevel.NONE,
                            'quarantined': False,
                            'health_score': 1.0
                        }
                
                # Save to database
                self._save_isolation_boundary(boundary)
                
                self.isolation_stats['boundaries_created'] += 1
                
                logger.info(f"Created isolation boundary: {boundary_id} ({boundary_type.value}, level {isolation_level.value})")
                return boundary_id
                
        except Exception as e:
            logger.error(f"Failed to create isolation boundary: {e}")
            return ""
    
    def isolate_component(self, component_name: str, trigger: IsolationTrigger,
                         isolation_level: Optional[IsolationLevel] = None,
                         duration_minutes: Optional[int] = None,
                         context: Optional[Dict[str, Any]] = None) -> str:
        """Isolate a component due to errors or security concerns."""
        try:
            event_id = f"isolation_{component_name}_{int(time.time() * 1000000)}"
            
            # Determine isolation level if not provided
            if isolation_level is None:
                isolation_level = self._determine_isolation_level(component_name, trigger)
            
            with self.isolation_lock:
                # Find applicable boundaries
                applicable_boundaries = [
                    boundary_id for boundary_id in self.component_boundaries.get(component_name, set())
                ]
                
                if not applicable_boundaries:
                    # Create emergency boundary
                    boundary_id = self.create_isolation_boundary(
                        BoundaryType.COMPONENT,
                        isolation_level,
                        {component_name}
                    )
                    applicable_boundaries = [boundary_id]
                
                # Execute isolation for each boundary
                isolation_success = True
                for boundary_id in applicable_boundaries:
                    success = self._execute_component_isolation(
                        boundary_id, component_name, isolation_level, trigger
                    )
                    isolation_success = isolation_success and success
                
                # Create isolation event
                isolation_event = IsolationEvent(
                    event_id=event_id,
                    boundary_id=applicable_boundaries[0] if applicable_boundaries else "",
                    trigger=trigger,
                    timestamp=datetime.now(),
                    source_component=component_name,
                    target_components=[component_name],
                    isolation_action=f"isolate_to_level_{isolation_level.value}",
                    success=isolation_success,
                    impact_assessment=self._assess_isolation_impact(component_name, isolation_level),
                    context=context or {}
                )
                
                # Store event
                self.isolation_events.append(isolation_event)
                self.isolation_stats['isolations_triggered'] += 1
                
                if isolation_success:
                    self.isolation_stats['successful_isolations'] += 1
                    
                    # Update component health
                    if component_name in self.component_health:
                        self.component_health[component_name]['isolation_level'] = isolation_level
                    
                    # Consider quarantine for severe cases
                    if isolation_level.value >= IsolationLevel.QUARANTINE.value and self.automatic_quarantine_enabled:
                        self._create_quarantine_zone(component_name, trigger, isolation_level)
                    
                else:
                    self.isolation_stats['failed_isolations'] += 1
                
                # Save to database
                self._save_isolation_event(isolation_event)
                
                logger.warning(f"Component isolation {'successful' if isolation_success else 'failed'}: {component_name} -> {isolation_level.name}")
                return event_id
                
        except Exception as e:
            logger.error(f"Error isolating component {component_name}: {e}")
            return ""
    
    def _execute_component_isolation(self, boundary_id: str, component_name: str,
                                   isolation_level: IsolationLevel, trigger: IsolationTrigger) -> bool:
        """Execute isolation for a specific component within a boundary."""
        try:
            boundary = self.isolation_boundaries.get(boundary_id)
            if not boundary:
                logger.error(f"Boundary not found: {boundary_id}")
                return False
            
            # Apply isolation based on level
            if isolation_level == IsolationLevel.SOFT:
                success = self._apply_soft_isolation(component_name, boundary)
            elif isolation_level == IsolationLevel.MODERATE:
                success = self._apply_moderate_isolation(component_name, boundary)
            elif isolation_level == IsolationLevel.STRICT:
                success = self._apply_strict_isolation(component_name, boundary)
            elif isolation_level == IsolationLevel.QUARANTINE:
                success = self._apply_quarantine_isolation(component_name, boundary)
            elif isolation_level == IsolationLevel.EMERGENCY:
                success = self._apply_emergency_isolation(component_name, boundary)
            else:
                logger.warning(f"Unknown isolation level: {isolation_level}")
                success = False
            
            if success:
                # Update boundary
                boundary.blocked_components.add(component_name)
                boundary.isolation_level = max(boundary.isolation_level, isolation_level)
                
                # Record interaction blocking
                self._block_component_interactions(component_name, boundary)
            
            return success
            
        except Exception as e:
            logger.error(f"Component isolation execution failed: {e}")
            return False
    
    def _apply_soft_isolation(self, component_name: str, boundary: IsolationBoundary) -> bool:
        """Apply soft isolation - monitoring and warnings only."""
        try:
            # Enable enhanced monitoring
            monitoring_rule = {
                'component': component_name,
                'level': 'enhanced',
                'metrics': ['error_rate', 'response_time', 'resource_usage'],
                'alert_threshold': 0.1
            }
            boundary.monitoring_rules.append(monitoring_rule)
            
            logger.info(f"Soft isolation applied to {component_name}: enhanced monitoring enabled")
            return True
            
        except Exception as e:
            logger.error(f"Soft isolation failed for {component_name}: {e}")
            return False
    
    def _apply_moderate_isolation(self, component_name: str, boundary: IsolationBoundary) -> bool:
        """Apply moderate isolation - limited interactions."""
        try:
            # Limit interactions to essential only
            essential_interactions = self._get_essential_interactions(component_name)
            boundary.allowed_interactions[component_name] = essential_interactions
            
            # Apply resource limits
            if not boundary.resource_limits:
                boundary.resource_limits = {}
            
            boundary.resource_limits[component_name] = {
                'max_memory': '512MB',
                'max_cpu': 0.5,
                'max_connections': 10,
                'rate_limit': 100
            }
            
            logger.info(f"Moderate isolation applied to {component_name}: limited interactions and resources")
            return True
            
        except Exception as e:
            logger.error(f"Moderate isolation failed for {component_name}: {e}")
            return False
    
    def _apply_strict_isolation(self, component_name: str, boundary: IsolationBoundary) -> bool:
        """Apply strict isolation - minimal interactions only."""
        try:
            # Block most interactions
            critical_interactions = self._get_critical_interactions(component_name)
            boundary.allowed_interactions[component_name] = critical_interactions
            
            # Strict resource limits
            boundary.resource_limits[component_name] = {
                'max_memory': '256MB',
                'max_cpu': 0.25,
                'max_connections': 5,
                'rate_limit': 50,
                'network_isolation': True
            }
            
            # Remove from non-critical services
            self._remove_from_load_balancers(component_name)
            
            logger.warning(f"Strict isolation applied to {component_name}: minimal interactions only")
            return True
            
        except Exception as e:
            logger.error(f"Strict isolation failed for {component_name}: {e}")
            return False
    
    def _apply_quarantine_isolation(self, component_name: str, boundary: IsolationBoundary) -> bool:
        """Apply quarantine isolation - complete isolation."""
        try:
            # Complete isolation - no external interactions
            boundary.allowed_interactions[component_name] = []
            
            # Minimal resource allocation
            boundary.resource_limits[component_name] = {
                'max_memory': '128MB',
                'max_cpu': 0.1,
                'max_connections': 1,
                'rate_limit': 10,
                'network_isolation': True,
                'disk_write_disabled': True
            }
            
            # Stop accepting new requests
            self._stop_request_routing(component_name)
            
            logger.critical(f"Quarantine isolation applied to {component_name}: complete isolation activated")
            return True
            
        except Exception as e:
            logger.error(f"Quarantine isolation failed for {component_name}: {e}")
            return False
    
    def _apply_emergency_isolation(self, component_name: str, boundary: IsolationBoundary) -> bool:
        """Apply emergency isolation - immediate shutdown."""
        try:
            # Immediate isolation and shutdown
            boundary.allowed_interactions[component_name] = []
            boundary.resource_limits[component_name] = {
                'max_memory': '64MB',
                'max_cpu': 0.05,
                'max_connections': 0,
                'rate_limit': 0,
                'shutdown_initiated': True
            }
            
            # Emergency actions
            self._initiate_emergency_shutdown(component_name)
            self._notify_administrators(component_name, "emergency_isolation")
            
            logger.critical(f"Emergency isolation applied to {component_name}: immediate shutdown initiated")
            return True
            
        except Exception as e:
            logger.error(f"Emergency isolation failed for {component_name}: {e}")
            return False
    
    def _create_quarantine_zone(self, component_name: str, trigger: IsolationTrigger,
                              isolation_level: IsolationLevel) -> str:
        """Create a quarantine zone for a problematic component."""
        try:
            zone_id = f"quarantine_{component_name}_{uuid.uuid4().hex[:8]}"
            
            # Define release conditions based on trigger
            release_conditions = self._get_quarantine_release_conditions(trigger)
            
            quarantine_zone = QuarantineZone(
                zone_id=zone_id,
                quarantined_components={component_name},
                quarantine_reason=f"{trigger.value}_triggered",
                quarantine_start=datetime.now(),
                isolation_level=isolation_level,
                monitoring_interval=30,  # Monitor every 30 seconds
                release_conditions=release_conditions,
                automatic_release=True,
                max_quarantine_duration=self.quarantine_duration_hours * 60  # Convert to minutes
            )
            
            # Store quarantine zone
            self.quarantine_zones[zone_id] = quarantine_zone
            
            # Update component health
            if component_name in self.component_health:
                self.component_health[component_name]['quarantined'] = True
                self.component_health[component_name]['quarantine_zone'] = zone_id
            
            # Save to database
            self._save_quarantine_zone(quarantine_zone)
            
            self.isolation_stats['quarantine_actions'] += 1
            
            logger.critical(f"Quarantine zone created: {zone_id} for {component_name}")
            return zone_id
            
        except Exception as e:
            logger.error(f"Failed to create quarantine zone for {component_name}: {e}")
            return ""
    
    def release_from_isolation(self, component_name: str, 
                             target_level: IsolationLevel = IsolationLevel.NONE) -> bool:
        """Release a component from isolation."""
        try:
            with self.isolation_lock:
                # Find component boundaries
                applicable_boundaries = list(self.component_boundaries.get(component_name, set()))
                
                if not applicable_boundaries:
                    logger.warning(f"No isolation boundaries found for {component_name}")
                    return True  # Already not isolated
                
                release_success = True
                
                # Release from each boundary
                for boundary_id in applicable_boundaries:
                    boundary = self.isolation_boundaries.get(boundary_id)
                    if boundary:
                        success = self._release_from_boundary(component_name, boundary, target_level)
                        release_success = release_success and success
                
                # Remove from quarantine if applicable
                quarantine_zone = self._find_component_quarantine_zone(component_name)
                if quarantine_zone:
                    self._release_from_quarantine(component_name, quarantine_zone)
                
                # Update component health
                if component_name in self.component_health:
                    self.component_health[component_name]['isolation_level'] = target_level
                    self.component_health[component_name]['quarantined'] = False
                
                if release_success:
                    self.isolation_stats['automatic_releases'] += 1
                    logger.info(f"Component released from isolation: {component_name}")
                else:
                    logger.error(f"Failed to release component from isolation: {component_name}")
                
                return release_success
                
        except Exception as e:
            logger.error(f"Error releasing component {component_name} from isolation: {e}")
            return False
    
    def _boundary_monitoring_loop(self):
        """Background boundary monitoring and violation detection loop."""
        while self.isolation_active:
            try:
                time.sleep(30)  # Monitor every 30 seconds
                
                with self.isolation_lock:
                    # Monitor all boundaries
                    for boundary_id, boundary in self.isolation_boundaries.items():
                        violations = self._check_boundary_violations(boundary)
                        
                        if violations:
                            self._handle_boundary_violations(boundary_id, violations)
                    
                    # Clean up old violations
                    self._cleanup_old_violations()
                
            except Exception as e:
                logger.error(f"Boundary monitoring loop error: {e}")
                time.sleep(60)
    
    def _cascade_detection_loop(self):
        """Background cascade detection and prevention loop."""
        while self.isolation_active:
            try:
                time.sleep(45)  # Check every 45 seconds
                
                with self.isolation_lock:
                    # Detect potential cascading failures
                    cascade_risks = self._detect_cascade_patterns()
                    
                    for risk in cascade_risks:
                        prevented = self._prevent_cascade_failure(risk)
                        if prevented:
                            self.isolation_stats['cascade_preventions'] += 1
                
            except Exception as e:
                logger.error(f"Cascade detection loop error: {e}")
                time.sleep(90)
    
    def get_comprehensive_isolation_report(self) -> Dict[str, Any]:
        """Generate comprehensive error isolation report."""
        with self.isolation_lock:
            # Boundary status summary
            boundary_summary = {}
            for boundary_id, boundary in self.isolation_boundaries.items():
                boundary_summary[boundary_id] = {
                    'type': boundary.boundary_type.value,
                    'isolation_level': boundary.isolation_level.name,
                    'protected_components': len(boundary.protected_components),
                    'blocked_components': len(boundary.blocked_components),
                    'violation_count': boundary.violation_count,
                    'last_violation': boundary.last_violation.isoformat() if boundary.last_violation else None
                }
            
            # Quarantine zones summary
            quarantine_summary = {}
            for zone_id, zone in self.quarantine_zones.items():
                quarantine_summary[zone_id] = {
                    'quarantined_components': len(zone.quarantined_components),
                    'quarantine_reason': zone.quarantine_reason,
                    'quarantine_duration_minutes': (datetime.now() - zone.quarantine_start).total_seconds() / 60,
                    'isolation_level': zone.isolation_level.name,
                    'automatic_release': zone.automatic_release,
                    'release_conditions': len(zone.release_conditions)
                }
            
            # Component health summary
            component_health_summary = {}
            for component, health in self.component_health.items():
                component_health_summary[component] = {
                    'error_count': health['error_count'],
                    'health_score': health['health_score'],
                    'isolation_level': health['isolation_level'].name if hasattr(health['isolation_level'], 'name') else str(health['isolation_level']),
                    'quarantined': health['quarantined'],
                    'last_error': health['last_error'].isoformat() if health['last_error'] else None
                }
            
            # Recent activity
            recent_cutoff = datetime.now() - timedelta(hours=1)
            recent_isolations = [
                event for event in self.isolation_events
                if event.timestamp >= recent_cutoff
            ]
            
            # Violation analysis
            total_violations = sum(len(violations) for violations in self.boundary_violations.values())
            
            return {
                'isolation_overview': {
                    'active_boundaries': len(self.isolation_boundaries),
                    'quarantine_zones': len(self.quarantine_zones),
                    'monitored_components': len(self.component_health),
                    'total_violations': total_violations,
                    'automatic_quarantine_enabled': self.automatic_quarantine_enabled
                },
                'isolation_statistics': self.isolation_stats.copy(),
                'boundary_status': boundary_summary,
                'quarantine_zones': quarantine_summary,
                'component_health': component_health_summary,
                'recent_activity': {
                    'isolations_last_hour': len(recent_isolations),
                    'cascade_detection_window': self.cascade_detection_window,
                    'boundary_violation_threshold': self.boundary_violation_threshold
                },
                'configuration': {
                    'max_isolation_depth': self.max_isolation_depth,
                    'quarantine_duration_hours': self.quarantine_duration_hours,
                    'cascade_detection_window': self.cascade_detection_window
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def shutdown(self):
        """Shutdown error isolation system."""
        self.isolation_active = False
        
        # Wait for threads to complete
        for thread in [self.boundary_monitor_thread, self.cascade_detector_thread,
                      self.quarantine_manager_thread, self.violation_analyzer_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info(f"Error Isolation System shutdown - Final Stats: {self.isolation_stats}")

# Global error isolation system instance
error_isolation_system = None