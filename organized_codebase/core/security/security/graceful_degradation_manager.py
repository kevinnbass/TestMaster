"""
Graceful Degradation Manager - Archive-Derived Robustness System
===============================================================

Intelligent graceful degradation system with adaptive performance
reduction, feature prioritization, and seamless quality adjustment.

Author: Agent C Security Framework
Created: 2025-08-21
"""

import logging
import time
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import statistics

logger = logging.getLogger(__name__)

class DegradationLevel(Enum):
    """Graceful degradation levels."""
    OPTIMAL = 0      # 100% functionality
    MINIMAL = 1      # 90% functionality - minor reductions
    LIGHT = 2        # 75% functionality - noticeable but acceptable
    MODERATE = 3     # 60% functionality - significant reduction
    HEAVY = 4        # 40% functionality - major limitations
    SURVIVAL = 5     # 20% functionality - essential only
    EMERGENCY = 6    # 5% functionality - absolute minimum

class DegradationTrigger(Enum):
    """Events that trigger degradation."""
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    HIGH_ERROR_RATE = "high_error_rate"
    PERFORMANCE_DECLINE = "performance_decline"
    DEPENDENCY_FAILURE = "dependency_failure"
    OVERLOAD_CONDITION = "overload_condition"
    MANUAL_REQUEST = "manual_request"
    PREDICTIVE_ACTION = "predictive_action"
    EMERGENCY_MODE = "emergency_mode"

class FeaturePriority(Enum):
    """Feature priority classifications."""
    CRITICAL = 1     # Never degrade - essential for operation
    HIGH = 2         # Degrade only under severe conditions
    NORMAL = 3       # Standard degradation candidate
    LOW = 4          # First to be degraded
    OPTIONAL = 5     # Can be completely disabled

@dataclass
class DegradationRule:
    """Graceful degradation rule definition."""
    rule_id: str
    component_pattern: str
    trigger_conditions: List[DegradationTrigger]
    degradation_steps: List[Dict[str, Any]]
    priority_overrides: Dict[str, FeaturePriority]
    resource_thresholds: Dict[str, float]
    recovery_conditions: Dict[str, Any]
    rollback_strategy: str
    monitoring_metrics: List[str]

@dataclass
class DegradationEvent:
    """Graceful degradation event record."""
    event_id: str
    component_name: str
    trigger: DegradationTrigger
    timestamp: datetime
    from_level: DegradationLevel
    to_level: DegradationLevel
    affected_features: List[str]
    performance_impact: Dict[str, float]
    user_impact_score: float
    recovery_estimate_minutes: int
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureState:
    """Current state of a system feature."""
    feature_name: str
    priority: FeaturePriority
    current_quality: float  # 0.0 to 1.0
    target_quality: float
    resource_usage: Dict[str, float]
    performance_metrics: Dict[str, float]
    degradation_history: List[Dict[str, Any]]
    recovery_callbacks: List[Callable]

class GracefulDegradationManager:
    """
    Advanced graceful degradation manager with intelligent feature prioritization.
    """
    
    def __init__(self, db_path: str = "data/graceful_degradation.db"):
        self.db_path = db_path
        
        # Initialize database
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
        # Component and feature tracking
        self.managed_components: Dict[str, Dict[str, Any]] = {}
        self.feature_states: Dict[str, FeatureState] = {}
        self.degradation_rules: Dict[str, DegradationRule] = {}
        
        # Current system state
        self.current_degradation_levels: Dict[str, DegradationLevel] = {}
        self.active_degradations: Dict[str, DegradationEvent] = {}
        self.resource_utilization: Dict[str, float] = defaultdict(float)
        
        # Degradation history and analytics
        self.degradation_history: deque = deque(maxlen=5000)
        self.quality_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Quality management
        self.quality_targets: Dict[str, float] = {}
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = {}
        self.degradation_strategies: Dict[str, List[Callable]] = {}
        
        # Configuration parameters
        self.degradation_sensitivity = 0.7  # 0.0 = aggressive, 1.0 = conservative
        self.recovery_hysteresis = 0.2      # Prevent oscillation
        self.quality_monitoring_interval = 15.0
        self.automatic_recovery_enabled = True
        self.predictive_degradation_enabled = True
        
        # Statistics and metrics
        self.degradation_stats = {
            'degradations_triggered': 0,
            'successful_recoveries': 0,
            'quality_improvements': 0,
            'user_impact_events': 0,
            'resource_savings': 0.0,
            'performance_improvements': 0,
            'automatic_adjustments': 0,
            'manual_overrides': 0,
            'total_uptime_preserved': 0.0,
            'average_degradation_duration': 0.0
        }
        
        # Background processing
        self.degradation_active = True
        self.quality_monitor_thread = threading.Thread(target=self._quality_monitoring_loop, daemon=True)
        self.degradation_controller_thread = threading.Thread(target=self._degradation_control_loop, daemon=True)
        self.recovery_manager_thread = threading.Thread(target=self._recovery_management_loop, daemon=True)
        self.adaptive_optimizer_thread = threading.Thread(target=self._adaptive_optimization_loop, daemon=True)
        
        # Start background threads
        self.quality_monitor_thread.start()
        self.degradation_controller_thread.start()
        self.recovery_manager_thread.start()
        
        if self.predictive_degradation_enabled:
            self.adaptive_optimizer_thread.start()
        
        # Thread safety
        self.degradation_lock = threading.RLock()
        
        logger.info("Graceful Degradation Manager initialized with adaptive optimization")
    
    def _init_database(self):
        """Initialize graceful degradation database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Degradation events table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS degradation_events (
                        event_id TEXT PRIMARY KEY,
                        component_name TEXT NOT NULL,
                        trigger_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        from_level INTEGER DEFAULT 0,
                        to_level INTEGER NOT NULL,
                        affected_features TEXT,
                        performance_impact TEXT,
                        user_impact_score REAL DEFAULT 0.0,
                        recovery_estimate_minutes INTEGER DEFAULT 0,
                        context TEXT
                    )
                ''')
                
                # Feature states table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS feature_states (
                        feature_name TEXT PRIMARY KEY,
                        priority INTEGER NOT NULL,
                        current_quality REAL DEFAULT 1.0,
                        target_quality REAL DEFAULT 1.0,
                        resource_usage TEXT,
                        performance_metrics TEXT,
                        degradation_history TEXT,
                        last_updated TEXT NOT NULL
                    )
                ''')
                
                # Degradation rules table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS degradation_rules (
                        rule_id TEXT PRIMARY KEY,
                        component_pattern TEXT NOT NULL,
                        trigger_conditions TEXT NOT NULL,
                        degradation_steps TEXT NOT NULL,
                        priority_overrides TEXT,
                        resource_thresholds TEXT,
                        recovery_conditions TEXT,
                        rollback_strategy TEXT,
                        monitoring_metrics TEXT
                    )
                ''')
                
                # Quality metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        metric_id TEXT PRIMARY KEY,
                        component_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        value REAL NOT NULL,
                        baseline_value REAL,
                        quality_impact REAL DEFAULT 0.0
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_events_component ON degradation_events(component_name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON degradation_events(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_component ON quality_metrics(component_name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON quality_metrics(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Graceful degradation database initialization failed: {e}")
            raise
    
    def register_component_degradation(self, component_name: str, 
                                     feature_definitions: Dict[str, Dict[str, Any]],
                                     quality_targets: Optional[Dict[str, float]] = None) -> bool:
        """Register component for graceful degradation management."""
        try:
            with self.degradation_lock:
                # Initialize component tracking
                self.managed_components[component_name] = {
                    'registered_at': datetime.now(),
                    'feature_count': len(feature_definitions),
                    'degradation_enabled': True,
                    'manual_override': False,
                    'baseline_performance': {}
                }
                
                # Initialize current degradation level
                self.current_degradation_levels[component_name] = DegradationLevel.OPTIMAL
                
                # Register features
                for feature_name, feature_config in feature_definitions.items():
                    priority = FeaturePriority(feature_config.get('priority', FeaturePriority.NORMAL.value))
                    
                    feature_state = FeatureState(
                        feature_name=feature_name,
                        priority=priority,
                        current_quality=1.0,
                        target_quality=1.0,
                        resource_usage=feature_config.get('resource_usage', {}),
                        performance_metrics={},
                        degradation_history=[],
                        recovery_callbacks=feature_config.get('recovery_callbacks', [])
                    )
                    
                    self.feature_states[f"{component_name}.{feature_name}"] = feature_state
                
                # Set quality targets
                if quality_targets:
                    for feature, target in quality_targets.items():
                        self.quality_targets[f"{component_name}.{feature}"] = target
                
                # Create default degradation rule
                self._create_default_degradation_rule(component_name, feature_definitions)
                
                logger.info(f"Registered component for graceful degradation: {component_name} ({len(feature_definitions)} features)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register component degradation for {component_name}: {e}")
            return False
    
    def trigger_degradation(self, component_name: str, trigger: DegradationTrigger,
                          target_level: Optional[DegradationLevel] = None,
                          context: Optional[Dict[str, Any]] = None) -> str:
        """Manually trigger graceful degradation for a component."""
        try:
            event_id = f"degradation_{component_name}_{int(time.time() * 1000000)}"
            
            # Determine current and target levels
            current_level = self.current_degradation_levels.get(component_name, DegradationLevel.OPTIMAL)
            
            if target_level is None:
                # Auto-determine target level based on trigger
                target_level = self._determine_target_degradation_level(component_name, trigger, current_level)
            
            with self.degradation_lock:
                # Create degradation event
                degradation_event = DegradationEvent(
                    event_id=event_id,
                    component_name=component_name,
                    trigger=trigger,
                    timestamp=datetime.now(),
                    from_level=current_level,
                    to_level=target_level,
                    affected_features=self._get_affected_features(component_name, target_level),
                    performance_impact={},
                    user_impact_score=self._calculate_user_impact(component_name, current_level, target_level),
                    recovery_estimate_minutes=self._estimate_recovery_time(trigger),
                    context=context or {}
                )
                
                # Execute degradation
                success = self._execute_degradation(degradation_event)
                
                if success:
                    # Store active degradation
                    self.active_degradations[event_id] = degradation_event
                    self.degradation_stats['degradations_triggered'] += 1
                    
                    # Update component level
                    self.current_degradation_levels[component_name] = target_level
                    
                    # Save to database
                    self._save_degradation_event(degradation_event)
                    
                    logger.info(f"Degradation triggered: {component_name} -> {target_level.name} ({event_id})")
                    return event_id
                else:
                    logger.error(f"Failed to execute degradation for {component_name}")
                    return ""
                
        except Exception as e:
            logger.error(f"Error triggering degradation for {component_name}: {e}")
            return ""
    
    def _execute_degradation(self, event: DegradationEvent) -> bool:
        """Execute graceful degradation steps for a component."""
        try:
            component_name = event.component_name
            target_level = event.to_level
            
            # Get applicable degradation rule
            rule = self._get_applicable_degradation_rule(component_name)
            
            if not rule:
                logger.warning(f"No degradation rule found for {component_name}")
                return False
            
            # Execute degradation steps based on target level
            degradation_steps = self._get_degradation_steps_for_level(rule, target_level)
            
            performance_impact = {}
            affected_features = []
            
            for step in degradation_steps:
                step_type = step.get('type', 'unknown')
                
                if step_type == 'reduce_quality':
                    success = self._reduce_feature_quality(component_name, step)
                elif step_type == 'disable_features':
                    success = self._disable_features(component_name, step)
                elif step_type == 'limit_resources':
                    success = self._limit_resource_usage(component_name, step)
                elif step_type == 'reduce_complexity':
                    success = self._reduce_processing_complexity(component_name, step)
                elif step_type == 'enable_caching':
                    success = self._enable_aggressive_caching(component_name, step)
                elif step_type == 'batch_operations':
                    success = self._enable_operation_batching(component_name, step)
                else:
                    logger.warning(f"Unknown degradation step type: {step_type}")
                    success = False
                
                if not success:
                    logger.error(f"Degradation step failed: {step_type} for {component_name}")
                    return False
                
                # Track impact
                feature_name = step.get('feature', 'unknown')
                impact = step.get('impact', 0.0)
                performance_impact[feature_name] = impact
                
                if feature_name not in affected_features:
                    affected_features.append(feature_name)
            
            # Update event with actual impact
            event.affected_features = affected_features
            event.performance_impact = performance_impact
            
            logger.info(f"Degradation executed successfully for {component_name}: {len(degradation_steps)} steps")
            return True
            
        except Exception as e:
            logger.error(f"Degradation execution failed for {event.component_name}: {e}")
            return False
    
    def _reduce_feature_quality(self, component_name: str, step: Dict[str, Any]) -> bool:
        """Reduce quality of specific features."""
        try:
            feature_name = step.get('feature')
            quality_reduction = step.get('quality_reduction', 0.2)
            
            feature_key = f"{component_name}.{feature_name}"
            
            if feature_key in self.feature_states:
                feature_state = self.feature_states[feature_key]
                new_quality = max(0.1, feature_state.current_quality - quality_reduction)
                feature_state.current_quality = new_quality
                
                # Record in history
                feature_state.degradation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'quality_reduction',
                    'from_quality': feature_state.current_quality + quality_reduction,
                    'to_quality': new_quality,
                    'reason': step.get('reason', 'degradation')
                })
                
                logger.debug(f"Reduced quality for {feature_key}: {new_quality:.2%}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Quality reduction failed: {e}")
            return False
    
    def _disable_features(self, component_name: str, step: Dict[str, Any]) -> bool:
        """Disable non-essential features."""
        try:
            features_to_disable = step.get('features', [])
            min_priority = FeaturePriority(step.get('min_priority', FeaturePriority.LOW.value))
            
            disabled_count = 0
            
            for feature_name in features_to_disable:
                feature_key = f"{component_name}.{feature_name}"
                
                if feature_key in self.feature_states:
                    feature_state = self.feature_states[feature_key]
                    
                    # Only disable if priority allows
                    if feature_state.priority.value >= min_priority.value:
                        feature_state.current_quality = 0.0
                        feature_state.degradation_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'action': 'feature_disabled',
                            'reason': 'graceful_degradation'
                        })
                        disabled_count += 1
                        logger.debug(f"Disabled feature: {feature_key}")
            
            return disabled_count > 0
            
        except Exception as e:
            logger.error(f"Feature disabling failed: {e}")
            return False
    
    def _limit_resource_usage(self, component_name: str, step: Dict[str, Any]) -> bool:
        """Limit resource usage for the component."""
        try:
            resource_limits = step.get('limits', {})
            
            for resource_type, limit in resource_limits.items():
                current_usage = self.resource_utilization.get(f"{component_name}.{resource_type}", 0.0)
                
                if current_usage > limit:
                    # Apply resource limitation
                    self.resource_utilization[f"{component_name}.{resource_type}"] = limit
                    logger.debug(f"Limited {resource_type} usage for {component_name}: {limit}")
            
            return True
            
        except Exception as e:
            logger.error(f"Resource limiting failed: {e}")
            return False
    
    def _reduce_processing_complexity(self, component_name: str, step: Dict[str, Any]) -> bool:
        """Reduce processing complexity to save resources."""
        try:
            complexity_reduction = step.get('reduction_factor', 0.5)
            affected_operations = step.get('operations', [])
            
            # Simulate complexity reduction
            for operation in affected_operations:
                operation_key = f"{component_name}.{operation}"
                
                # Reduce operation complexity
                self.resource_utilization[f"{operation_key}.complexity"] = complexity_reduction
                logger.debug(f"Reduced complexity for {operation_key}: {complexity_reduction}")
            
            return True
            
        except Exception as e:
            logger.error(f"Complexity reduction failed: {e}")
            return False
    
    def _enable_aggressive_caching(self, component_name: str, step: Dict[str, Any]) -> bool:
        """Enable aggressive caching to reduce load."""
        try:
            cache_settings = step.get('cache_settings', {})
            
            # Simulate enabling aggressive caching
            cache_key = f"{component_name}.caching"
            self.resource_utilization[cache_key] = cache_settings.get('aggressiveness', 0.8)
            
            logger.debug(f"Enabled aggressive caching for {component_name}")
            return True
            
        except Exception as e:
            logger.error(f"Caching enablement failed: {e}")
            return False
    
    def _enable_operation_batching(self, component_name: str, step: Dict[str, Any]) -> bool:
        """Enable operation batching to improve efficiency."""
        try:
            batch_size = step.get('batch_size', 10)
            batch_timeout = step.get('timeout_ms', 100)
            
            # Simulate operation batching
            batch_key = f"{component_name}.batching"
            self.resource_utilization[batch_key] = batch_size / 100.0  # Normalize
            
            logger.debug(f"Enabled operation batching for {component_name}: size={batch_size}")
            return True
            
        except Exception as e:
            logger.error(f"Operation batching failed: {e}")
            return False
    
    def initiate_recovery(self, component_name: str, target_level: DegradationLevel = DegradationLevel.OPTIMAL) -> bool:
        """Initiate recovery from degraded state."""
        try:
            with self.degradation_lock:
                current_level = self.current_degradation_levels.get(component_name, DegradationLevel.OPTIMAL)
                
                if current_level.value <= target_level.value:
                    logger.info(f"Component {component_name} already at or above target level")
                    return True
                
                # Execute recovery steps
                recovery_success = self._execute_recovery_steps(component_name, current_level, target_level)
                
                if recovery_success:
                    # Update degradation level
                    self.current_degradation_levels[component_name] = target_level
                    self.degradation_stats['successful_recoveries'] += 1
                    
                    # Remove from active degradations if fully recovered
                    if target_level == DegradationLevel.OPTIMAL:
                        events_to_remove = [
                            event_id for event_id, event in self.active_degradations.items()
                            if event.component_name == component_name
                        ]
                        
                        for event_id in events_to_remove:
                            del self.active_degradations[event_id]
                    
                    logger.info(f"Recovery successful for {component_name}: {current_level.name} -> {target_level.name}")
                    return True
                else:
                    logger.error(f"Recovery failed for {component_name}")
                    return False
                
        except Exception as e:
            logger.error(f"Recovery initiation failed for {component_name}: {e}")
            return False
    
    def _quality_monitoring_loop(self):
        """Background quality monitoring and adjustment loop."""
        while self.degradation_active:
            try:
                time.sleep(self.quality_monitoring_interval)
                
                with self.degradation_lock:
                    # Monitor quality metrics for all components
                    for component_name in self.managed_components.keys():
                        self._monitor_component_quality(component_name)
                    
                    # Check for automatic recovery opportunities
                    if self.automatic_recovery_enabled:
                        self._check_recovery_opportunities()
                    
                    # Update performance baselines
                    self._update_performance_baselines()
                
            except Exception as e:
                logger.error(f"Quality monitoring loop error: {e}")
                time.sleep(30)
    
    def get_comprehensive_degradation_report(self) -> Dict[str, Any]:
        """Generate comprehensive graceful degradation report."""
        with self.degradation_lock:
            # Component status summary
            component_status = {}
            for component_name, component_info in self.managed_components.items():
                current_level = self.current_degradation_levels.get(component_name, DegradationLevel.OPTIMAL)
                
                # Calculate overall quality
                component_features = [
                    f for f in self.feature_states.keys()
                    if f.startswith(f"{component_name}.")
                ]
                
                if component_features:
                    avg_quality = statistics.mean([
                        self.feature_states[f].current_quality for f in component_features
                    ])
                else:
                    avg_quality = 1.0
                
                component_status[component_name] = {
                    'degradation_level': current_level.name,
                    'level_value': current_level.value,
                    'average_quality': avg_quality,
                    'feature_count': len(component_features),
                    'degradation_enabled': component_info.get('degradation_enabled', True),
                    'manual_override': component_info.get('manual_override', False)
                }
            
            # Active degradations summary
            active_degradations_summary = {}
            for event_id, event in self.active_degradations.items():
                active_degradations_summary[event_id] = {
                    'component': event.component_name,
                    'trigger': event.trigger.value,
                    'level': event.to_level.name,
                    'duration_minutes': (datetime.now() - event.timestamp).total_seconds() / 60,
                    'affected_features': len(event.affected_features),
                    'user_impact_score': event.user_impact_score
                }
            
            # Feature states summary
            feature_summary = {}
            for feature_key, feature_state in self.feature_states.items():
                feature_summary[feature_key] = {
                    'priority': feature_state.priority.name,
                    'current_quality': feature_state.current_quality,
                    'target_quality': feature_state.target_quality,
                    'degradation_events': len(feature_state.degradation_history)
                }
            
            # Recent degradation activity
            recent_cutoff = datetime.now() - timedelta(hours=1)
            recent_degradations = [
                event for event in self.degradation_history
                if event.timestamp >= recent_cutoff
            ]
            
            return {
                'degradation_overview': {
                    'managed_components': len(self.managed_components),
                    'active_degradations': len(self.active_degradations),
                    'total_features': len(self.feature_states),
                    'degradation_rules': len(self.degradation_rules),
                    'automatic_recovery_enabled': self.automatic_recovery_enabled,
                    'predictive_degradation_enabled': self.predictive_degradation_enabled
                },
                'component_status': component_status,
                'active_degradations': active_degradations_summary,
                'feature_states': feature_summary,
                'degradation_statistics': self.degradation_stats.copy(),
                'recent_activity': {
                    'degradations_last_hour': len(recent_degradations),
                    'quality_monitoring_interval': self.quality_monitoring_interval,
                    'degradation_sensitivity': self.degradation_sensitivity,
                    'recovery_hysteresis': self.recovery_hysteresis
                },
                'resource_utilization': dict(self.resource_utilization),
                'quality_targets': self.quality_targets.copy(),
                'timestamp': datetime.now().isoformat()
            }
    
    def shutdown(self):
        """Shutdown graceful degradation manager."""
        self.degradation_active = False
        
        # Wait for threads to complete
        for thread in [self.quality_monitor_thread, self.degradation_controller_thread,
                      self.recovery_manager_thread, self.adaptive_optimizer_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info(f"Graceful Degradation Manager shutdown - Final Stats: {self.degradation_stats}")

# Global graceful degradation manager instance
graceful_degradation_manager = None