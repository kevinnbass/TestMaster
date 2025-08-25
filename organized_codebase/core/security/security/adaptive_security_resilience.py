"""
Archive Derived Adaptive Security Resilience Module
Extracted from TestMaster archive quantum retry systems for security resilience
Enhanced for adaptive security responses, failure recovery, and threat resilience
"""

import uuid
import time
import json
import math
import random
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from .error_handler import SecurityError, security_error_handler


class ResilienceStrategy(Enum):
    """Security resilience strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_RECOVERY = "linear_recovery"
    ADAPTIVE_LEARNING = "adaptive_learning"
    QUANTUM_RESILIENCE = "quantum_resilience"
    PREDICTIVE_RECOVERY = "predictive_recovery"
    NEURAL_ADAPTATION = "neural_adaptation"


class SecurityFailurePattern(Enum):
    """Security failure pattern types"""
    TRANSIENT_ATTACK = "transient_attack"
    PERSISTENT_THREAT = "persistent_threat"
    CASCADING_FAILURE = "cascading_failure"
    PERIODIC_INTRUSION = "periodic_intrusion"
    RANDOM_ANOMALY = "random_anomaly"
    COORDINATED_ATTACK = "coordinated_attack"


class RecoveryPriority(Enum):
    """Recovery priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class SecurityComponent(Enum):
    """Security components that can fail"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    FIREWALL = "firewall"
    INTRUSION_DETECTION = "intrusion_detection"
    VULNERABILITY_SCANNER = "vulnerability_scanner"
    AUDIT_LOGGING = "audit_logging"
    BACKUP_SYSTEM = "backup_system"


@dataclass
class SecurityFailure:
    """Security system failure record"""
    failure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    component: SecurityComponent = SecurityComponent.AUTHENTICATION
    failure_type: str = ""
    description: str = ""
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    severity: str = "medium"
    error_details: Dict[str, Any] = field(default_factory=dict)
    pattern: SecurityFailurePattern = SecurityFailurePattern.RANDOM_ANOMALY
    recovery_attempts: int = 0
    recovered_at: Optional[datetime] = None
    recovery_duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert failure to dictionary"""
        return {
            'failure_id': self.failure_id,
            'component': self.component.value,
            'failure_type': self.failure_type,
            'description': self.description,
            'occurred_at': self.occurred_at.isoformat(),
            'severity': self.severity,
            'error_details': self.error_details,
            'pattern': self.pattern.value,
            'recovery_attempts': self.recovery_attempts,
            'recovered_at': self.recovered_at.isoformat() if self.recovered_at else None,
            'recovery_duration': self.recovery_duration
        }


@dataclass
class RecoveryAction:
    """Security recovery action definition"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_name: str = ""
    component: SecurityComponent = SecurityComponent.AUTHENTICATION
    recovery_function: Optional[Callable] = None
    max_attempts: int = 5
    priority: RecoveryPriority = RecoveryPriority.NORMAL
    success_rate: float = 0.0
    average_recovery_time: float = 0.0
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResilienceMetrics:
    """Security resilience metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_failures: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    average_recovery_time: float = 0.0
    resilience_score: float = 100.0
    component_health: Dict[str, float] = field(default_factory=dict)
    threat_resistance: float = 100.0
    adaptation_effectiveness: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_failures': self.total_failures,
            'successful_recoveries': self.successful_recoveries,
            'failed_recoveries': self.failed_recoveries,
            'average_recovery_time': self.average_recovery_time,
            'resilience_score': self.resilience_score,
            'component_health': self.component_health,
            'threat_resistance': self.threat_resistance,
            'adaptation_effectiveness': self.adaptation_effectiveness
        }


class FailurePatternAnalyzer:
    """Advanced failure pattern analysis and prediction"""
    
    def __init__(self):
        self.failure_history: deque = deque(maxlen=1000)
        self.pattern_signatures: Dict[SecurityFailurePattern, List[str]] = {}
        self.pattern_probabilities: Dict[SecurityFailurePattern, float] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize pattern signatures
        self._initialize_pattern_signatures()
    
    def _initialize_pattern_signatures(self):
        """Initialize failure pattern signatures"""
        self.pattern_signatures = {
            SecurityFailurePattern.TRANSIENT_ATTACK: [
                'network_timeout', 'connection_refused', 'temporary_unavailable'
            ],
            SecurityFailurePattern.PERSISTENT_THREAT: [
                'repeated_failed_auth', 'continuous_scanning', 'sustained_attack'
            ],
            SecurityFailurePattern.CASCADING_FAILURE: [
                'dependency_failure', 'service_unavailable', 'resource_exhausted'
            ],
            SecurityFailurePattern.PERIODIC_INTRUSION: [
                'scheduled_attack', 'time_based_pattern', 'regular_intervals'
            ],
            SecurityFailurePattern.COORDINATED_ATTACK: [
                'multiple_sources', 'synchronized_timing', 'distributed_attack'
            ]
        }
    
    def analyze_failure_pattern(self, failure: SecurityFailure) -> SecurityFailurePattern:
        """Analyze failure to determine pattern"""
        try:
            # Check against known pattern signatures
            error_text = failure.description.lower() + " " + str(failure.error_details).lower()
            
            pattern_scores = {}
            
            for pattern, signatures in self.pattern_signatures.items():
                score = 0
                for signature in signatures:
                    if signature in error_text:
                        score += 1
                
                pattern_scores[pattern] = score / len(signatures) if signatures else 0
            
            # Find best matching pattern
            if pattern_scores:
                best_pattern = max(pattern_scores, key=pattern_scores.get)
                if pattern_scores[best_pattern] > 0.3:  # Minimum confidence threshold
                    return best_pattern
            
            # Analyze temporal patterns
            recent_failures = [f for f in self.failure_history if 
                             (datetime.utcnow() - f.occurred_at).total_seconds() < 3600]
            
            if len(recent_failures) >= 3:
                # Check if failures are in regular intervals (periodic)
                intervals = []
                for i in range(1, len(recent_failures)):
                    interval = (recent_failures[i].occurred_at - recent_failures[i-1].occurred_at).total_seconds()
                    intervals.append(interval)
                
                if len(intervals) >= 2:
                    # Check for periodic pattern
                    avg_interval = sum(intervals) / len(intervals)
                    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                    
                    if variance < (avg_interval * 0.2) ** 2:  # Low variance indicates periodicity
                        return SecurityFailurePattern.PERIODIC_INTRUSION
            
            # Default to random anomaly if no pattern detected
            return SecurityFailurePattern.RANDOM_ANOMALY
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return SecurityFailurePattern.RANDOM_ANOMALY
    
    def predict_next_failure(self, component: SecurityComponent) -> Tuple[float, datetime]:
        """Predict probability and timing of next failure"""
        try:
            component_failures = [
                f for f in self.failure_history 
                if f.component == component and
                (datetime.utcnow() - f.occurred_at).total_seconds() < 86400  # Last 24 hours
            ]
            
            if len(component_failures) < 2:
                return 0.1, datetime.utcnow() + timedelta(days=1)  # Low probability, distant future
            
            # Calculate failure rate
            time_span = (datetime.utcnow() - component_failures[0].occurred_at).total_seconds()
            failure_rate = len(component_failures) / (time_span / 3600)  # failures per hour
            
            # Predict next failure based on exponential distribution
            if failure_rate > 0:
                # Expected time to next failure (exponential distribution)
                expected_hours = 1 / failure_rate
                next_failure_time = datetime.utcnow() + timedelta(hours=expected_hours)
                
                # Probability increases over time
                probability = min(failure_rate * 0.1, 0.9)  # Cap at 90%
                
                return probability, next_failure_time
            
            return 0.1, datetime.utcnow() + timedelta(days=1)
            
        except Exception as e:
            self.logger.error(f"Failure prediction failed: {e}")
            return 0.1, datetime.utcnow() + timedelta(days=1)
    
    def add_failure(self, failure: SecurityFailure):
        """Add failure to history for analysis"""
        failure.pattern = self.analyze_failure_pattern(failure)
        self.failure_history.append(failure)
        
        # Update pattern probabilities
        pattern_counts = defaultdict(int)
        for f in self.failure_history:
            pattern_counts[f.pattern] += 1
        
        total_failures = len(self.failure_history)
        for pattern, count in pattern_counts.items():
            self.pattern_probabilities[pattern] = count / total_failures


class AdaptiveRecoveryEngine:
    """Adaptive recovery engine with machine learning capabilities"""
    
    def __init__(self):
        self.recovery_actions: Dict[SecurityComponent, List[RecoveryAction]] = defaultdict(list)
        self.success_history: Dict[str, List[bool]] = defaultdict(list)
        self.adaptation_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.learning_rate = 0.1
        self.logger = logging.getLogger(__name__)
        
        # Initialize default recovery actions
        self._initialize_recovery_actions()
    
    def _initialize_recovery_actions(self):
        """Initialize default recovery actions for each component"""
        # Authentication recovery actions
        auth_actions = [
            RecoveryAction(
                action_name="Restart Authentication Service",
                component=SecurityComponent.AUTHENTICATION,
                recovery_function=self._restart_auth_service,
                priority=RecoveryPriority.HIGH
            ),
            RecoveryAction(
                action_name="Clear Authentication Cache",
                component=SecurityComponent.AUTHENTICATION,
                recovery_function=self._clear_auth_cache,
                priority=RecoveryPriority.NORMAL
            ),
            RecoveryAction(
                action_name="Fallback to Backup Auth",
                component=SecurityComponent.AUTHENTICATION,
                recovery_function=self._fallback_auth,
                priority=RecoveryPriority.CRITICAL
            )
        ]
        self.recovery_actions[SecurityComponent.AUTHENTICATION] = auth_actions
        
        # Firewall recovery actions
        firewall_actions = [
            RecoveryAction(
                action_name="Reload Firewall Rules",
                component=SecurityComponent.FIREWALL,
                recovery_function=self._reload_firewall_rules,
                priority=RecoveryPriority.HIGH
            ),
            RecoveryAction(
                action_name="Reset Firewall Connections",
                component=SecurityComponent.FIREWALL,
                recovery_function=self._reset_firewall_connections,
                priority=RecoveryPriority.NORMAL
            )
        ]
        self.recovery_actions[SecurityComponent.FIREWALL] = firewall_actions
        
        # Add more default actions for other components...
    
    def select_recovery_action(self, failure: SecurityFailure) -> Optional[RecoveryAction]:
        """Select best recovery action using adaptive learning"""
        try:
            actions = self.recovery_actions.get(failure.component, [])
            if not actions:
                return None
            
            # Score actions based on success history and adaptation weights
            action_scores = {}
            
            for action in actions:
                # Base score from historical success rate
                base_score = action.success_rate
                
                # Adaptation weight (learned from experience)
                adaptation_score = self.adaptation_weights[action.action_id]
                
                # Priority bonus
                priority_bonus = (6 - action.priority.value) / 5.0  # Higher priority = higher score
                
                # Recency penalty (prefer actions that haven't been used recently)
                recency_penalty = 0
                if action.last_used:
                    hours_since_use = (datetime.utcnow() - action.last_used).total_seconds() / 3600
                    recency_penalty = max(0, 1 - (hours_since_use / 24))  # Penalty decreases over 24 hours
                
                # Combined score
                final_score = (base_score * 0.4 + adaptation_score * 0.4 + 
                             priority_bonus * 0.2) * (1 - recency_penalty * 0.1)
                
                action_scores[action] = final_score
            
            # Select action with highest score
            best_action = max(action_scores, key=action_scores.get)
            return best_action
            
        except Exception as e:
            self.logger.error(f"Recovery action selection failed: {e}")
            return None
    
    def execute_recovery_action(self, action: RecoveryAction, failure: SecurityFailure) -> bool:
        """Execute recovery action and update learning metrics"""
        try:
            start_time = time.time()
            
            # Execute the recovery function
            if action.recovery_function:
                success = action.recovery_function(failure)
            else:
                # Default recovery behavior
                success = self._default_recovery(failure)
            
            execution_time = time.time() - start_time
            
            # Update action metrics
            action.last_used = datetime.utcnow()
            
            # Update success history
            self.success_history[action.action_id].append(success)
            if len(self.success_history[action.action_id]) > 50:  # Keep last 50 attempts
                self.success_history[action.action_id] = self.success_history[action.action_id][-50:]
            
            # Recalculate success rate
            recent_successes = self.success_history[action.action_id]
            action.success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
            
            # Update average recovery time
            if action.average_recovery_time == 0:
                action.average_recovery_time = execution_time
            else:
                action.average_recovery_time = (action.average_recovery_time * 0.8 + execution_time * 0.2)
            
            # Adaptive learning: update weights based on success/failure
            if success:
                # Reward successful actions
                self.adaptation_weights[action.action_id] = min(2.0, 
                    self.adaptation_weights[action.action_id] * (1 + self.learning_rate))
            else:
                # Penalize failed actions
                self.adaptation_weights[action.action_id] = max(0.1, 
                    self.adaptation_weights[action.action_id] * (1 - self.learning_rate))
            
            self.logger.info(f"Recovery action executed: {action.action_name} - Success: {success}")
            return success
            
        except Exception as e:
            self.logger.error(f"Recovery action execution failed: {e}")
            return False
    
    def _restart_auth_service(self, failure: SecurityFailure) -> bool:
        """Restart authentication service"""
        try:
            # Simulate service restart
            time.sleep(0.1)  # Simulate restart time
            self.logger.info("Authentication service restarted")
            return True
        except Exception as e:
            self.logger.error(f"Auth service restart failed: {e}")
            return False
    
    def _clear_auth_cache(self, failure: SecurityFailure) -> bool:
        """Clear authentication cache"""
        try:
            # Simulate cache clearing
            self.logger.info("Authentication cache cleared")
            return True
        except Exception as e:
            self.logger.error(f"Auth cache clear failed: {e}")
            return False
    
    def _fallback_auth(self, failure: SecurityFailure) -> bool:
        """Fallback to backup authentication system"""
        try:
            # Simulate fallback activation
            self.logger.info("Backup authentication system activated")
            return True
        except Exception as e:
            self.logger.error(f"Auth fallback failed: {e}")
            return False
    
    def _reload_firewall_rules(self, failure: SecurityFailure) -> bool:
        """Reload firewall rules"""
        try:
            # Simulate firewall rule reload
            time.sleep(0.2)
            self.logger.info("Firewall rules reloaded")
            return True
        except Exception as e:
            self.logger.error(f"Firewall rule reload failed: {e}")
            return False
    
    def _reset_firewall_connections(self, failure: SecurityFailure) -> bool:
        """Reset firewall connections"""
        try:
            # Simulate connection reset
            self.logger.info("Firewall connections reset")
            return True
        except Exception as e:
            self.logger.error(f"Firewall connection reset failed: {e}")
            return False
    
    def _default_recovery(self, failure: SecurityFailure) -> bool:
        """Default recovery action"""
        try:
            self.logger.info(f"Default recovery attempted for {failure.component.value}")
            return random.choice([True, False])  # Simulate 50% success rate
        except Exception:
            return False


class AdaptiveSecurityResilienceManager:
    """Comprehensive adaptive security resilience management system"""
    
    def __init__(self):
        # Core components
        self.pattern_analyzer = FailurePatternAnalyzer()
        self.recovery_engine = AdaptiveRecoveryEngine()
        
        # Data storage
        self.active_failures: Dict[str, SecurityFailure] = {}
        self.recovery_history: deque = deque(maxlen=1000)
        self.component_health: Dict[SecurityComponent, float] = {}
        
        # Initialize component health
        for component in SecurityComponent:
            self.component_health[component] = 100.0
        
        # Statistics
        self.stats = {
            'total_failures': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'resilience_score': 100.0,
            'adaptation_effectiveness': 100.0,
            'threat_resistance': 100.0
        }
        
        # Configuration
        self.auto_recovery_enabled = True
        self.max_recovery_attempts = 5
        self.health_decay_rate = 0.1  # Health decreases by this amount per failure
        self.health_recovery_rate = 0.05  # Health increases by this amount per successful recovery
        
        # Background processing
        self.resilience_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        
        # Thread safety
        self.resilience_lock = threading.RLock()
        
        # Start background threads
        self.monitor_thread.start()
        self.health_thread.start()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Adaptive Security Resilience Manager initialized")
    
    def report_failure(self, component: SecurityComponent, failure_type: str,
                      description: str, severity: str = "medium",
                      error_details: Dict[str, Any] = None) -> str:
        """Report security component failure"""
        try:
            with self.resilience_lock:
                failure = SecurityFailure(
                    component=component,
                    failure_type=failure_type,
                    description=description,
                    severity=severity,
                    error_details=error_details or {}
                )
                
                # Analyze failure pattern
                self.pattern_analyzer.add_failure(failure)
                
                # Store active failure
                self.active_failures[failure.failure_id] = failure
                
                # Update statistics
                self.stats['total_failures'] += 1
                
                # Decrease component health
                current_health = self.component_health[component]
                health_impact = {'low': 5, 'medium': 10, 'high': 20, 'critical': 40}.get(severity, 10)
                self.component_health[component] = max(0, current_health - health_impact)
                
                # Trigger auto-recovery if enabled
                if self.auto_recovery_enabled:
                    threading.Thread(
                        target=self._attempt_recovery,
                        args=(failure,),
                        daemon=True
                    ).start()
                
                self.logger.warning(f"Security failure reported: {component.value} - {description}")
                return failure.failure_id
                
        except Exception as e:
            error = SecurityError(f"Failed to report security failure: {str(e)}", "RESILIENCE_001")
            security_error_handler.handle_error(error)
            return ""
    
    def _attempt_recovery(self, failure: SecurityFailure):
        """Attempt to recover from security failure"""
        try:
            recovery_start = time.time()
            
            for attempt in range(self.max_recovery_attempts):
                # Select recovery action
                action = self.recovery_engine.select_recovery_action(failure)
                if not action:
                    self.logger.warning(f"No recovery action available for {failure.component.value}")
                    break
                
                # Execute recovery action
                success = self.recovery_engine.execute_recovery_action(action, failure)
                
                failure.recovery_attempts += 1
                
                if success:
                    # Recovery successful
                    failure.recovered_at = datetime.utcnow()
                    failure.recovery_duration = time.time() - recovery_start
                    
                    # Update statistics
                    with self.resilience_lock:
                        self.stats['successful_recoveries'] += 1
                        
                        # Update average recovery time
                        if self.stats['average_recovery_time'] == 0:
                            self.stats['average_recovery_time'] = failure.recovery_duration
                        else:
                            self.stats['average_recovery_time'] = (
                                self.stats['average_recovery_time'] * 0.8 + 
                                failure.recovery_duration * 0.2
                            )
                        
                        # Improve component health
                        current_health = self.component_health[failure.component]
                        health_improvement = {'low': 2, 'medium': 5, 'high': 10, 'critical': 20}.get(failure.severity, 5)
                        self.component_health[failure.component] = min(100, current_health + health_improvement)
                    
                    # Remove from active failures
                    if failure.failure_id in self.active_failures:
                        del self.active_failures[failure.failure_id]
                    
                    # Add to recovery history
                    self.recovery_history.append(failure)
                    
                    self.logger.info(f"Security failure recovered: {failure.failure_id}")
                    return
                else:
                    # Recovery failed, try again after backoff
                    backoff_time = min(300, 2 ** attempt)  # Exponential backoff, max 5 minutes
                    time.sleep(backoff_time)
            
            # All recovery attempts failed
            with self.resilience_lock:
                self.stats['failed_recoveries'] += 1
                
                # Significantly decrease component health
                current_health = self.component_health[failure.component]
                self.component_health[failure.component] = max(0, current_health - 30)
            
            self.logger.error(f"Security failure recovery failed: {failure.failure_id}")
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
    
    def get_resilience_metrics(self) -> ResilienceMetrics:
        """Get current resilience metrics"""
        with self.resilience_lock:
            # Calculate resilience score
            total_attempts = self.stats['successful_recoveries'] + self.stats['failed_recoveries']
            if total_attempts > 0:
                success_rate = self.stats['successful_recoveries'] / total_attempts
                self.stats['resilience_score'] = success_rate * 100
            
            # Calculate component health scores
            component_health_dict = {
                component.value: health 
                for component, health in self.component_health.items()
            }
            
            # Calculate threat resistance (based on recent failure frequency)
            recent_failures = [
                f for f in self.pattern_analyzer.failure_history
                if (datetime.utcnow() - f.occurred_at).total_seconds() < 3600
            ]
            threat_resistance = max(0, 100 - len(recent_failures) * 5)
            
            # Calculate adaptation effectiveness
            adaptation_weights = list(self.recovery_engine.adaptation_weights.values())
            if adaptation_weights:
                avg_adaptation = sum(adaptation_weights) / len(adaptation_weights)
                adaptation_effectiveness = min(100, avg_adaptation * 50)
            else:
                adaptation_effectiveness = 100
            
            return ResilienceMetrics(
                total_failures=self.stats['total_failures'],
                successful_recoveries=self.stats['successful_recoveries'],
                failed_recoveries=self.stats['failed_recoveries'],
                average_recovery_time=self.stats['average_recovery_time'],
                resilience_score=self.stats['resilience_score'],
                component_health=component_health_dict,
                threat_resistance=threat_resistance,
                adaptation_effectiveness=adaptation_effectiveness
            )
    
    def predict_component_failure(self, component: SecurityComponent) -> Tuple[float, datetime]:
        """Predict next failure for component"""
        return self.pattern_analyzer.predict_next_failure(component)
    
    def get_component_health(self, component: SecurityComponent) -> float:
        """Get health score for specific component"""
        return self.component_health.get(component, 0.0)
    
    def _monitoring_loop(self):
        """Background monitoring for proactive resilience"""
        while self.resilience_active:
            try:
                time.sleep(300)  # Check every 5 minutes
                
                with self.resilience_lock:
                    # Check for components with low health
                    for component, health in self.component_health.items():
                        if health < 50:  # Component health below 50%
                            self.logger.warning(f"Low component health detected: {component.value} ({health:.1f}%)")
                            
                            # Predict next failure
                            failure_prob, predicted_time = self.predict_component_failure(component)
                            if failure_prob > 0.7:  # High probability of failure
                                self.logger.warning(f"High failure probability for {component.value}: {failure_prob:.2f}")
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    def _health_monitoring_loop(self):
        """Background health monitoring and gradual recovery"""
        while self.resilience_active:
            try:
                time.sleep(60)  # Check every minute
                
                with self.resilience_lock:
                    # Gradually improve health of stable components
                    for component in SecurityComponent:
                        current_health = self.component_health[component]
                        
                        # Check if component has been stable (no recent failures)
                        recent_failures = [
                            f for f in self.pattern_analyzer.failure_history
                            if f.component == component and
                            (datetime.utcnow() - f.occurred_at).total_seconds() < 1800  # Last 30 minutes
                        ]
                        
                        if not recent_failures and current_health < 100:
                            # Gradually restore health
                            self.component_health[component] = min(100, current_health + self.health_recovery_rate)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def shutdown(self):
        """Shutdown resilience manager"""
        self.resilience_active = False
        self.logger.info("Adaptive Security Resilience Manager shutdown")


# Global adaptive security resilience manager
adaptive_security_resilience = AdaptiveSecurityResilienceManager()


def report_security_failure(component: SecurityComponent, failure_type: str,
                           description: str, severity: str = "medium",
                           error_details: Dict[str, Any] = None) -> str:
    """Convenience function to report security failure"""
    return adaptive_security_resilience.report_failure(
        component, failure_type, description, severity, error_details
    )


def get_component_health(component: SecurityComponent) -> float:
    """Convenience function to get component health"""
    return adaptive_security_resilience.get_component_health(component)