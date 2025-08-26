"""
Self-Healing Coordinator - Archive-Derived Robustness System
==========================================================

Intelligent self-healing system with automatic failure detection,
diagnosis, and recovery coordination across all system components.

Author: Agent C Security Framework
Created: 2025-08-21
"""

import logging
import time
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
import os
import random
import statistics

logger = logging.getLogger(__name__)

class SystemHealth(Enum):
    """System-wide health status levels."""
    OPTIMAL = "optimal"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RecoveryStrategy(Enum):
    """Self-healing recovery strategies."""
    IMMEDIATE_RESTART = "immediate_restart"
    GRACEFUL_RESTART = "graceful_restart"
    COMPONENT_ISOLATION = "component_isolation"
    FAILOVER_SWITCH = "failover_switch"
    DEGRADED_MODE = "degraded_mode"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    SELF_REPAIR = "self_repair"
    CASCADING_RECOVERY = "cascading_recovery"

class HealingPriority(Enum):
    """Self-healing priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class ComponentFailure:
    """Detailed component failure record."""
    failure_id: str
    component_name: str
    timestamp: datetime
    failure_type: str
    error_message: str
    stack_trace: str
    severity: int
    affected_dependencies: List[str]
    recovery_suggested: RecoveryStrategy
    context: Dict[str, Any] = field(default_factory=dict)
    cascade_potential: float = 0.0

@dataclass
class HealingAction:
    """Self-healing action record."""
    action_id: str
    component_name: str
    strategy: RecoveryStrategy
    timestamp: datetime
    priority: HealingPriority
    estimated_duration: int
    success_probability: float
    prerequisite_actions: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)

@dataclass
class ComponentStatus:
    """Comprehensive component status tracking."""
    component_name: str
    health: SystemHealth
    last_check: datetime
    failure_count: int
    recovery_count: int
    uptime_percentage: float
    response_time_ms: float
    error_rate: float
    dependencies_healthy: bool
    backup_available: bool
    last_recovery: Optional[datetime] = None
    healing_history: List[str] = field(default_factory=list)

class SelfHealingCoordinator:
    """
    Advanced self-healing coordinator with predictive failure detection.
    """
    
    def __init__(self, db_path: str = "data/self_healing.db"):
        self.db_path = db_path
        
        # Initialize database
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
        # Component registry and tracking
        self.registered_components: Dict[str, Dict[str, Any]] = {}
        self.component_statuses: Dict[str, ComponentStatus] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        self.backup_components: Dict[str, List[str]] = {}
        
        # Failure and healing tracking
        self.failure_history: deque = deque(maxlen=10000)
        self.healing_actions: deque = deque(maxlen=5000)
        self.active_healings: Dict[str, HealingAction] = {}
        
        # Predictive analytics
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.healing_effectiveness: Dict[RecoveryStrategy, Dict[str, float]] = defaultdict(lambda: {
            'success_rate': 0.0,
            'average_duration': 0.0,
            'total_attempts': 0,
            'successful_attempts': 0
        })
        
        # Health monitoring
        self.system_health = SystemHealth.OPTIMAL
        self.health_check_interval = 15.0  # seconds
        self.failure_prediction_window = 300  # 5 minutes
        
        # Configuration parameters
        self.cascade_threshold = 0.7
        self.emergency_threshold = 0.9
        self.auto_healing_enabled = True
        self.predictive_healing_enabled = True
        
        # Statistics
        self.healing_stats = {
            'total_failures_detected': 0,
            'healing_actions_taken': 0,
            'successful_healings': 0,
            'failed_healings': 0,
            'cascades_prevented': 0,
            'emergencies_resolved': 0,
            'predictive_healings': 0,
            'uptime_improvements': 0.0,
            'mean_time_to_healing': 0.0,
            'healing_success_rate': 100.0
        }
        
        # Background processing
        self.healing_active = True
        self.health_monitor_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        self.healing_coordinator_thread = threading.Thread(target=self._healing_coordination_loop, daemon=True)
        self.failure_predictor_thread = threading.Thread(target=self._failure_prediction_loop, daemon=True)
        self.cascade_detector_thread = threading.Thread(target=self._cascade_detection_loop, daemon=True)
        
        # Start background threads
        self.health_monitor_thread.start()
        self.healing_coordinator_thread.start()
        
        if self.predictive_healing_enabled:
            self.failure_predictor_thread.start()
            self.cascade_detector_thread.start()
        
        # Thread safety
        self.healing_lock = threading.RLock()
        
        logger.info("Self-Healing Coordinator initialized with predictive capabilities")
    
    def _init_database(self):
        """Initialize self-healing database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Component failures table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS component_failures (
                        failure_id TEXT PRIMARY KEY,
                        component_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        failure_type TEXT NOT NULL,
                        error_message TEXT,
                        stack_trace TEXT,
                        severity INTEGER DEFAULT 3,
                        affected_dependencies TEXT,
                        recovery_suggested TEXT,
                        context TEXT,
                        cascade_potential REAL DEFAULT 0.0
                    )
                ''')
                
                # Healing actions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS healing_actions (
                        action_id TEXT PRIMARY KEY,
                        component_name TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        estimated_duration INTEGER DEFAULT 0,
                        success_probability REAL DEFAULT 0.5,
                        prerequisite_actions TEXT,
                        affected_systems TEXT,
                        success INTEGER DEFAULT 0,
                        actual_duration REAL DEFAULT 0.0
                    )
                ''')
                
                # Component status table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS component_status (
                        component_name TEXT PRIMARY KEY,
                        health TEXT NOT NULL,
                        last_check TEXT NOT NULL,
                        failure_count INTEGER DEFAULT 0,
                        recovery_count INTEGER DEFAULT 0,
                        uptime_percentage REAL DEFAULT 100.0,
                        response_time_ms REAL DEFAULT 0.0,
                        error_rate REAL DEFAULT 0.0,
                        dependencies_healthy INTEGER DEFAULT 1,
                        backup_available INTEGER DEFAULT 0,
                        last_recovery TEXT,
                        healing_history TEXT
                    )
                ''')
                
                # Healing patterns table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS healing_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        component_pattern TEXT NOT NULL,
                        failure_pattern TEXT NOT NULL,
                        healing_strategy TEXT NOT NULL,
                        success_rate REAL NOT NULL,
                        confidence REAL NOT NULL,
                        discovered_at TEXT NOT NULL,
                        usage_count INTEGER DEFAULT 0
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_failures_component ON component_failures(component_name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_failures_timestamp ON component_failures(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_actions_component ON healing_actions(component_name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_status_health ON component_status(health)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Self-healing database initialization failed: {e}")
            raise
    
    def register_component(self, component_name: str, component_instance: Any,
                          health_check: Callable[[], Dict[str, Any]],
                          dependencies: Optional[List[str]] = None,
                          backup_components: Optional[List[str]] = None) -> bool:
        """Register a component for self-healing monitoring."""
        try:
            with self.healing_lock:
                self.registered_components[component_name] = {
                    'instance': component_instance,
                    'health_check': health_check,
                    'registered_at': datetime.now(),
                    'healing_enabled': True
                }
                
                # Initialize component status
                self.component_statuses[component_name] = ComponentStatus(
                    component_name=component_name,
                    health=SystemHealth.OPTIMAL,
                    last_check=datetime.now(),
                    failure_count=0,
                    recovery_count=0,
                    uptime_percentage=100.0,
                    response_time_ms=0.0,
                    error_rate=0.0,
                    dependencies_healthy=True,
                    backup_available=bool(backup_components)
                )
                
                # Store dependencies
                if dependencies:
                    self.component_dependencies[component_name] = dependencies
                
                # Store backup components
                if backup_components:
                    self.backup_components[component_name] = backup_components
                
                # Save to database
                self._save_component_status(self.component_statuses[component_name])
                
                logger.info(f"Registered component for self-healing: {component_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register component {component_name}: {e}")
            return False
    
    def report_failure(self, component_name: str, error: Exception, 
                      context: Optional[Dict[str, Any]] = None) -> str:
        """Report a component failure for immediate healing assessment."""
        try:
            failure_id = f"failure_{component_name}_{int(time.time() * 1000000)}"
            
            # Analyze error severity and cascade potential
            severity = self._analyze_error_severity(error)
            cascade_potential = self._calculate_cascade_potential(component_name, error)
            
            # Create failure record
            failure = ComponentFailure(
                failure_id=failure_id,
                component_name=component_name,
                timestamp=datetime.now(),
                failure_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                severity=severity,
                affected_dependencies=self._get_affected_dependencies(component_name),
                recovery_suggested=self._suggest_recovery_strategy(component_name, error),
                context=context or {},
                cascade_potential=cascade_potential
            )
            
            with self.healing_lock:
                # Store failure
                self.failure_history.append(failure)
                self.healing_stats['total_failures_detected'] += 1
                
                # Update component status
                if component_name in self.component_statuses:
                    status = self.component_statuses[component_name]
                    status.failure_count += 1
                    status.health = self._determine_health_from_severity(severity)
                    status.last_check = datetime.now()
                    
                    # Update error rate
                    self._update_component_error_rate(component_name)
                
                # Save to database
                self._save_failure_record(failure)
                
                # Trigger immediate healing if enabled
                if self.auto_healing_enabled:
                    self._initiate_healing(failure)
                
                logger.warning(f"Failure reported for {component_name}: {failure_id}")
                return failure_id
                
        except Exception as e:
            logger.error(f"Error reporting failure for {component_name}: {e}")
            return ""
    
    def initiate_manual_healing(self, component_name: str, 
                               strategy: RecoveryStrategy = RecoveryStrategy.SELF_REPAIR) -> bool:
        """Manually initiate healing for a component."""
        try:
            with self.healing_lock:
                if component_name not in self.registered_components:
                    logger.error(f"Component not registered: {component_name}")
                    return False
                
                # Create healing action
                action = HealingAction(
                    action_id=f"manual_{component_name}_{int(time.time() * 1000000)}",
                    component_name=component_name,
                    strategy=strategy,
                    timestamp=datetime.now(),
                    priority=HealingPriority.HIGH,
                    estimated_duration=self._estimate_healing_duration(strategy),
                    success_probability=self._calculate_healing_success_probability(component_name, strategy),
                    affected_systems=self._get_affected_systems(component_name)
                )
                
                # Execute healing
                return self._execute_healing_action(action)
                
        except Exception as e:
            logger.error(f"Manual healing failed for {component_name}: {e}")
            return False
    
    def _health_monitoring_loop(self):
        """Continuous health monitoring loop."""
        while self.healing_active:
            try:
                cycle_start = time.time()
                
                with self.healing_lock:
                    # Check all registered components
                    for component_name, component_info in self.registered_components.items():
                        if not component_info.get('healing_enabled', True):
                            continue
                        
                        try:
                            # Execute health check
                            health_result = component_info['health_check']()
                            self._process_health_check_result(component_name, health_result)
                            
                        except Exception as e:
                            # Health check failed - report as failure
                            self.report_failure(component_name, e, {'health_check_failure': True})
                    
                    # Update system-wide health
                    self._update_system_health()
                
                # Adaptive sleep based on system health
                cycle_duration = time.time() - cycle_start
                sleep_time = max(5.0, min(self.health_check_interval, self.health_check_interval - cycle_duration))
                
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                time.sleep(30)
    
    def _healing_coordination_loop(self):
        """Healing coordination and execution loop."""
        while self.healing_active:
            try:
                time.sleep(5)  # Check for healing actions every 5 seconds
                
                with self.healing_lock:
                    # Process queued healing actions
                    self._process_healing_queue()
                    
                    # Check for completed healings
                    self._check_healing_completion()
                    
                    # Analyze healing effectiveness
                    self._analyze_healing_effectiveness()
                
            except Exception as e:
                logger.error(f"Healing coordination loop error: {e}")
                time.sleep(10)
    
    def _failure_prediction_loop(self):
        """Predictive failure detection loop."""
        while self.healing_active:
            try:
                time.sleep(60)  # Analyze patterns every minute
                
                with self.healing_lock:
                    # Analyze failure patterns for predictions
                    predictions = self._analyze_failure_patterns()
                    
                    # Execute predictive healing for high-risk components
                    for component_name, risk_score in predictions.items():
                        if risk_score > 0.7:  # High risk threshold
                            self._execute_predictive_healing(component_name, risk_score)
                            self.healing_stats['predictive_healings'] += 1
                
            except Exception as e:
                logger.error(f"Failure prediction loop error: {e}")
                time.sleep(120)
    
    def _cascade_detection_loop(self):
        """Cascade failure detection and prevention loop."""
        while self.healing_active:
            try:
                time.sleep(30)  # Check for cascade risks every 30 seconds
                
                with self.healing_lock:
                    cascade_risks = self._detect_cascade_risks()
                    
                    if cascade_risks:
                        prevented = self._prevent_cascade_failures(cascade_risks)
                        self.healing_stats['cascades_prevented'] += prevented
                
            except Exception as e:
                logger.error(f"Cascade detection loop error: {e}")
                time.sleep(60)
    
    def _initiate_healing(self, failure: ComponentFailure):
        """Initiate healing action for a reported failure."""
        try:
            # Determine healing priority based on severity and cascade potential
            if failure.severity <= 2 or failure.cascade_potential > self.cascade_threshold:
                priority = HealingPriority.CRITICAL
            elif failure.severity <= 3:
                priority = HealingPriority.HIGH
            else:
                priority = HealingPriority.NORMAL
            
            # Create healing action
            action = HealingAction(
                action_id=f"healing_{failure.failure_id}",
                component_name=failure.component_name,
                strategy=failure.recovery_suggested,
                timestamp=datetime.now(),
                priority=priority,
                estimated_duration=self._estimate_healing_duration(failure.recovery_suggested),
                success_probability=self._calculate_healing_success_probability(
                    failure.component_name, failure.recovery_suggested
                ),
                affected_systems=failure.affected_dependencies
            )
            
            # Queue for execution
            self.active_healings[action.action_id] = action
            self.healing_stats['healing_actions_taken'] += 1
            
            logger.info(f"Healing initiated for {failure.component_name}: {action.action_id}")
            
            # Execute immediately if critical
            if priority == HealingPriority.CRITICAL:
                self._execute_healing_action(action)
            
        except Exception as e:
            logger.error(f"Error initiating healing: {e}")
    
    def _execute_healing_action(self, action: HealingAction) -> bool:
        """Execute a specific healing action."""
        try:
            healing_start = time.time()
            
            logger.info(f"Executing healing: {action.action_id} ({action.strategy.value})")
            
            component_name = action.component_name
            component_info = self.registered_components.get(component_name)
            
            if not component_info:
                logger.error(f"Component not found for healing: {component_name}")
                return False
            
            component_instance = component_info['instance']
            success = False
            
            # Execute strategy-specific healing
            if action.strategy == RecoveryStrategy.IMMEDIATE_RESTART:
                success = self._execute_immediate_restart(component_instance)
            elif action.strategy == RecoveryStrategy.GRACEFUL_RESTART:
                success = self._execute_graceful_restart(component_instance)
            elif action.strategy == RecoveryStrategy.COMPONENT_ISOLATION:
                success = self._execute_component_isolation(component_name)
            elif action.strategy == RecoveryStrategy.FAILOVER_SWITCH:
                success = self._execute_failover_switch(component_name)
            elif action.strategy == RecoveryStrategy.DEGRADED_MODE:
                success = self._execute_degraded_mode(component_instance)
            elif action.strategy == RecoveryStrategy.SELF_REPAIR:
                success = self._execute_self_repair(component_instance)
            elif action.strategy == RecoveryStrategy.CASCADING_RECOVERY:
                success = self._execute_cascading_recovery(component_name)
            else:
                logger.warning(f"Unknown healing strategy: {action.strategy}")
                success = False
            
            healing_duration = time.time() - healing_start
            
            # Update statistics and effectiveness tracking
            strategy_stats = self.healing_effectiveness[action.strategy]
            strategy_stats['total_attempts'] += 1
            
            if success:
                strategy_stats['successful_attempts'] += 1
                self.healing_stats['successful_healings'] += 1
                
                # Update component status
                if component_name in self.component_statuses:
                    status = self.component_statuses[component_name]
                    status.recovery_count += 1
                    status.last_recovery = datetime.now()
                    status.health = SystemHealth.HEALTHY
                    status.healing_history.append(f"{action.strategy.value}:success")
            else:
                self.healing_stats['failed_healings'] += 1
                
                # Try fallback strategy
                fallback_success = self._try_fallback_healing(action)
                if fallback_success:
                    success = True
                    self.healing_stats['successful_healings'] += 1
            
            # Update effectiveness metrics
            strategy_stats['success_rate'] = (
                strategy_stats['successful_attempts'] / strategy_stats['total_attempts']
            )
            
            # Update mean time to healing
            total_healings = self.healing_stats['successful_healings'] + self.healing_stats['failed_healings']
            if total_healings > 0:
                current_mean = self.healing_stats['mean_time_to_healing']
                self.healing_stats['mean_time_to_healing'] = (
                    (current_mean * (total_healings - 1) + healing_duration) / total_healings
                )
            
            # Save healing action result
            self._save_healing_action(action, success, healing_duration)
            
            logger.info(f"Healing {'succeeded' if success else 'failed'} for {component_name} in {healing_duration:.2f}s")
            return success
            
        except Exception as e:
            logger.error(f"Healing execution failed for {action.action_id}: {e}")
            return False
        finally:
            # Remove from active healings
            self.active_healings.pop(action.action_id, None)
    
    def get_comprehensive_healing_report(self) -> Dict[str, Any]:
        """Generate comprehensive self-healing report."""
        with self.healing_lock:
            # Calculate system-wide statistics
            total_components = len(self.registered_components)
            healthy_components = sum(
                1 for status in self.component_statuses.values()
                if status.health in [SystemHealth.OPTIMAL, SystemHealth.HEALTHY]
            )
            
            # Calculate overall success rate
            total_healing_attempts = (
                self.healing_stats['successful_healings'] + 
                self.healing_stats['failed_healings']
            )
            
            if total_healing_attempts > 0:
                success_rate = (self.healing_stats['successful_healings'] / total_healing_attempts) * 100
            else:
                success_rate = 100.0
            
            # Get recent failures and healings
            recent_failures = [
                f for f in self.failure_history
                if (datetime.now() - f.timestamp).total_seconds() < 3600
            ]
            
            recent_healings = [
                a for a in self.healing_actions
                if (datetime.now() - a.timestamp).total_seconds() < 3600
            ]
            
            return {
                'system_health': {
                    'overall_status': self.system_health.value,
                    'healthy_components': healthy_components,
                    'total_components': total_components,
                    'health_percentage': (healthy_components / max(1, total_components)) * 100
                },
                'component_statuses': {
                    name: {
                        'health': status.health.value,
                        'uptime_percentage': status.uptime_percentage,
                        'failure_count': status.failure_count,
                        'recovery_count': status.recovery_count,
                        'response_time_ms': status.response_time_ms,
                        'error_rate': status.error_rate,
                        'backup_available': status.backup_available,
                        'last_recovery': status.last_recovery.isoformat() if status.last_recovery else None
                    }
                    for name, status in self.component_statuses.items()
                },
                'healing_statistics': self.healing_stats.copy(),
                'healing_effectiveness': {
                    strategy.value: {
                        'success_rate': stats['success_rate'] * 100,
                        'total_attempts': stats['total_attempts'],
                        'successful_attempts': stats['successful_attempts']
                    }
                    for strategy, stats in self.healing_effectiveness.items()
                },
                'recent_activity': {
                    'failures_last_hour': len(recent_failures),
                    'healings_last_hour': len(recent_healings),
                    'active_healings': len(self.active_healings)
                },
                'predictive_analysis': {
                    'patterns_detected': len(self.failure_patterns),
                    'predictive_healing_enabled': self.predictive_healing_enabled,
                    'cascade_detection_active': True
                },
                'configuration': {
                    'auto_healing_enabled': self.auto_healing_enabled,
                    'health_check_interval': self.health_check_interval,
                    'cascade_threshold': self.cascade_threshold,
                    'emergency_threshold': self.emergency_threshold
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def shutdown(self):
        """Shutdown self-healing coordinator."""
        self.healing_active = False
        
        # Wait for threads to complete
        for thread in [self.health_monitor_thread, self.healing_coordinator_thread,
                      self.failure_predictor_thread, self.cascade_detector_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info(f"Self-Healing Coordinator shutdown - Final Stats: {self.healing_stats}")

# Global self-healing coordinator instance
self_healing_coordinator = None