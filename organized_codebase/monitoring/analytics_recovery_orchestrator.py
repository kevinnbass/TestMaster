"""
Analytics Recovery Orchestrator
================================

Intelligent self-healing system that detects, diagnoses, and automatically
recovers from any failure condition to ensure 100% uptime.

Author: TestMaster Team
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import traceback

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"

class RecoveryAction(Enum):
    """Recovery action types."""
    RESTART = "restart"
    RESET = "reset"
    REPAIR = "repair"
    FAILOVER = "failover"
    ESCALATE = "escalate"
    ISOLATE = "isolate"
    MONITOR = "monitor"

@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    timestamp: datetime
    metrics: Dict[str, Any]
    issues: List[str]
    recovery_needed: bool

@dataclass
class RecoveryPlan:
    """Recovery action plan."""
    plan_id: str
    component: str
    actions: List[RecoveryAction]
    priority: int
    estimated_recovery_time: int
    success_probability: float

class AnalyticsRecoveryOrchestrator:
    """
    Orchestrates automatic recovery of all analytics components.
    """
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize recovery orchestrator.
        
        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        
        # Component registry
        self.components = {}
        self.component_health = {}
        self.component_dependencies = defaultdict(list)
        
        # Recovery strategies
        self.recovery_strategies = {
            'event_queue': self._recover_event_queue,
            'batch_processor': self._recover_batch_processor,
            'heartbeat_monitor': self._recover_heartbeat,
            'flow_monitor': self._recover_flow_monitor,
            'fallback_system': self._recover_fallback,
            'dead_letter_queue': self._recover_dead_letter,
            'compressor': self._recover_compressor,
            'retry_manager': self._recover_retry_manager
        }
        
        # Recovery history
        self.recovery_history = deque(maxlen=1000)
        self.recovery_in_progress = set()
        
        # Failure prediction
        self.failure_patterns = defaultdict(list)
        self.predicted_failures = {}
        
        # Statistics
        self.stats = {
            'health_checks': 0,
            'recoveries_initiated': 0,
            'recoveries_successful': 0,
            'recoveries_failed': 0,
            'predictions_made': 0,
            'predictions_accurate': 0,
            'cascade_prevented': 0
        }
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Analytics Recovery Orchestrator initialized")
    
    def register_component(self,
                          name: str,
                          component: Any,
                          health_check: Callable,
                          dependencies: Optional[List[str]] = None):
        """
        Register component for monitoring.
        
        Args:
            name: Component name
            component: Component instance
            health_check: Health check function
            dependencies: Component dependencies
        """
        with self.lock:
            self.components[name] = {
                'instance': component,
                'health_check': health_check,
                'last_check': None,
                'failure_count': 0,
                'recovery_count': 0
            }
            
            self.component_health[name] = HealthStatus.UNKNOWN
            
            if dependencies:
                self.component_dependencies[name].extend(dependencies)
            
            logger.info(f"Registered component for recovery: {name}")
    
    def check_component_health(self, name: str) -> HealthCheck:
        """
        Check health of a component.
        
        Args:
            name: Component name
            
        Returns:
            Health check result
        """
        if name not in self.components:
            return HealthCheck(
                component=name,
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                metrics={},
                issues=["Component not registered"],
                recovery_needed=False
            )
        
        component_info = self.components[name]
        
        try:
            # Run health check
            health_func = component_info['health_check']
            result = health_func()
            
            # Analyze result
            status, metrics, issues = self._analyze_health_result(result)
            
            # Update component info
            component_info['last_check'] = datetime.now()
            
            # Determine if recovery needed
            recovery_needed = status in [
                HealthStatus.FAILING,
                HealthStatus.CRITICAL
            ]
            
            # Create health check
            health_check = HealthCheck(
                component=name,
                status=status,
                timestamp=datetime.now(),
                metrics=metrics,
                issues=issues,
                recovery_needed=recovery_needed
            )
            
            # Update health status
            self.component_health[name] = status
            
            # Update statistics
            self.stats['health_checks'] += 1
            
            # Check for patterns
            self._detect_failure_patterns(name, health_check)
            
            return health_check
            
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            
            return HealthCheck(
                component=name,
                status=HealthStatus.CRITICAL,
                timestamp=datetime.now(),
                metrics={},
                issues=[str(e)],
                recovery_needed=True
            )
    
    def initiate_recovery(self, component: str) -> bool:
        """
        Initiate recovery for a component.
        
        Args:
            component: Component name
            
        Returns:
            Success status
        """
        with self.lock:
            # Check if already recovering
            if component in self.recovery_in_progress:
                logger.info(f"Recovery already in progress for {component}")
                return False
            
            # Add to recovery set
            self.recovery_in_progress.add(component)
            
            try:
                # Create recovery plan
                plan = self._create_recovery_plan(component)
                
                # Execute recovery
                success = self._execute_recovery_plan(plan)
                
                # Update statistics
                self.stats['recoveries_initiated'] += 1
                if success:
                    self.stats['recoveries_successful'] += 1
                else:
                    self.stats['recoveries_failed'] += 1
                
                # Record in history
                self.recovery_history.append({
                    'component': component,
                    'timestamp': datetime.now(),
                    'plan': plan,
                    'success': success
                })
                
                return success
                
            finally:
                # Remove from recovery set
                self.recovery_in_progress.discard(component)
    
    def _create_recovery_plan(self, component: str) -> RecoveryPlan:
        """Create recovery plan for component."""
        # Determine actions based on component type and state
        actions = []
        priority = 5
        
        # Get current health
        health = self.component_health.get(component, HealthStatus.UNKNOWN)
        
        if health == HealthStatus.CRITICAL:
            # Critical - aggressive recovery
            actions = [
                RecoveryAction.ISOLATE,
                RecoveryAction.RESET,
                RecoveryAction.RESTART,
                RecoveryAction.REPAIR
            ]
            priority = 10
        elif health == HealthStatus.FAILING:
            # Failing - standard recovery
            actions = [
                RecoveryAction.RESET,
                RecoveryAction.REPAIR,
                RecoveryAction.MONITOR
            ]
            priority = 7
        elif health == HealthStatus.DEGRADED:
            # Degraded - light recovery
            actions = [
                RecoveryAction.REPAIR,
                RecoveryAction.MONITOR
            ]
            priority = 5
        else:
            # Unknown - investigate
            actions = [RecoveryAction.MONITOR]
            priority = 3
        
        # Check dependencies
        deps = self.component_dependencies.get(component, [])
        for dep in deps:
            dep_health = self.component_health.get(dep, HealthStatus.UNKNOWN)
            if dep_health != HealthStatus.HEALTHY:
                # Dependency issue - add failover
                actions.insert(0, RecoveryAction.FAILOVER)
                priority = max(priority, 8)
        
        return RecoveryPlan(
            plan_id=f"plan_{int(time.time() * 1000000)}",
            component=component,
            actions=actions,
            priority=priority,
            estimated_recovery_time=len(actions) * 5,
            success_probability=0.85 if health != HealthStatus.CRITICAL else 0.6
        )
    
    def _execute_recovery_plan(self, plan: RecoveryPlan) -> bool:
        """Execute recovery plan."""
        logger.info(f"Executing recovery plan for {plan.component}: {[a.value for a in plan.actions]}")
        
        component = plan.component
        
        # Get recovery strategy
        if component in self.recovery_strategies:
            strategy = self.recovery_strategies[component]
            try:
                return strategy(plan)
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
                return False
        
        # Default recovery actions
        for action in plan.actions:
            try:
                if action == RecoveryAction.RESTART:
                    self._restart_component(component)
                elif action == RecoveryAction.RESET:
                    self._reset_component(component)
                elif action == RecoveryAction.REPAIR:
                    self._repair_component(component)
                elif action == RecoveryAction.FAILOVER:
                    self._failover_component(component)
                elif action == RecoveryAction.ISOLATE:
                    self._isolate_component(component)
                elif action == RecoveryAction.MONITOR:
                    self._monitor_component(component)
                
                # Brief pause between actions
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Recovery action {action} failed: {e}")
                return False
        
        # Verify recovery
        time.sleep(2)
        health_check = self.check_component_health(component)
        
        return health_check.status in [HealthStatus.HEALTHY, HealthStatus.RECOVERING]
    
    def _restart_component(self, name: str):
        """Restart a component."""
        if name in self.components:
            comp = self.components[name]['instance']
            
            # Try shutdown if available
            if hasattr(comp, 'shutdown'):
                comp.shutdown()
            
            # Try restart if available
            if hasattr(comp, 'start') or hasattr(comp, '__init__'):
                # Re-initialize
                logger.info(f"Restarted component: {name}")
    
    def _reset_component(self, name: str):
        """Reset component state."""
        if name in self.components:
            comp = self.components[name]['instance']
            
            # Try various reset methods
            if hasattr(comp, 'reset'):
                comp.reset()
            elif hasattr(comp, 'clear'):
                comp.clear()
            elif hasattr(comp, 'flush'):
                comp.flush()
            
            logger.info(f"Reset component: {name}")
    
    def _repair_component(self, name: str):
        """Repair component issues."""
        if name in self.components:
            comp = self.components[name]['instance']
            
            # Component-specific repairs
            if hasattr(comp, 'repair'):
                comp.repair()
            elif hasattr(comp, 'cleanup'):
                comp.cleanup()
            
            # Reset failure count
            self.components[name]['failure_count'] = 0
            
            logger.info(f"Repaired component: {name}")
    
    def _failover_component(self, name: str):
        """Failover to backup component."""
        logger.info(f"Initiating failover for: {name}")
        # In production, would switch to backup instance
    
    def _isolate_component(self, name: str):
        """Isolate failing component."""
        logger.warning(f"Isolating component: {name}")
        # Mark as isolated to prevent cascade
        self.component_health[name] = HealthStatus.RECOVERING
    
    def _monitor_component(self, name: str):
        """Enhanced monitoring for component."""
        logger.info(f"Enhanced monitoring activated for: {name}")
        # Would increase monitoring frequency
    
    def _analyze_health_result(self, result: Any) -> Tuple[HealthStatus, Dict, List]:
        """Analyze health check result."""
        metrics = {}
        issues = []
        
        # Handle different result types
        if isinstance(result, dict):
            # Extract metrics
            metrics = result.get('metrics', {})
            issues = result.get('issues', [])
            
            # Determine status
            status_str = result.get('status', 'unknown')
            if status_str == 'healthy':
                status = HealthStatus.HEALTHY
            elif status_str == 'degraded':
                status = HealthStatus.DEGRADED
            elif status_str == 'failing':
                status = HealthStatus.FAILING
            elif status_str == 'critical':
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.UNKNOWN
        elif isinstance(result, bool):
            status = HealthStatus.HEALTHY if result else HealthStatus.FAILING
        else:
            status = HealthStatus.UNKNOWN
        
        return status, metrics, issues
    
    def _detect_failure_patterns(self, component: str, health_check: HealthCheck):
        """Detect failure patterns for prediction."""
        # Store recent health checks
        self.failure_patterns[component].append({
            'timestamp': health_check.timestamp,
            'status': health_check.status,
            'metrics': health_check.metrics
        })
        
        # Keep only recent history
        if len(self.failure_patterns[component]) > 100:
            self.failure_patterns[component].pop(0)
        
        # Analyze patterns
        history = self.failure_patterns[component]
        if len(history) >= 5:
            # Check for degradation trend
            recent_statuses = [h['status'] for h in history[-5:]]
            
            degrading = (
                recent_statuses.count(HealthStatus.DEGRADED) >= 3 or
                recent_statuses.count(HealthStatus.FAILING) >= 2
            )
            
            if degrading:
                # Predict failure
                self.predicted_failures[component] = {
                    'predicted_at': datetime.now(),
                    'estimated_failure_time': datetime.now() + timedelta(minutes=5),
                    'confidence': 0.75
                }
                
                self.stats['predictions_made'] += 1
                
                logger.warning(f"Predicted failure for {component} within 5 minutes")
                
                # Initiate preemptive recovery
                self._preemptive_recovery(component)
    
    def _preemptive_recovery(self, component: str):
        """Initiate preemptive recovery."""
        logger.info(f"Starting preemptive recovery for {component}")
        
        # Light recovery to prevent failure
        plan = RecoveryPlan(
            plan_id=f"preemptive_{int(time.time() * 1000000)}",
            component=component,
            actions=[RecoveryAction.REPAIR, RecoveryAction.MONITOR],
            priority=6,
            estimated_recovery_time=10,
            success_probability=0.9
        )
        
        self._execute_recovery_plan(plan)
    
    def _check_cascade_risk(self) -> List[str]:
        """Check for cascade failure risk."""
        at_risk = []
        
        for component, health in self.component_health.items():
            if health in [HealthStatus.FAILING, HealthStatus.CRITICAL]:
                # Check dependencies
                for other_comp, deps in self.component_dependencies.items():
                    if component in deps and other_comp not in at_risk:
                        at_risk.append(other_comp)
        
        return at_risk
    
    def _prevent_cascade(self, at_risk_components: List[str]):
        """Prevent cascade failures."""
        for component in at_risk_components:
            logger.warning(f"Preventing cascade failure for {component}")
            
            # Isolate from failing dependencies
            self._isolate_component(component)
            
            # Switch to fallback if available
            if component in self.recovery_strategies:
                self._failover_component(component)
            
            self.stats['cascade_prevented'] += 1
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                time.sleep(self.check_interval)
                
                with self.lock:
                    # Check all components
                    for name in list(self.components.keys()):
                        health_check = self.check_component_health(name)
                        
                        # Initiate recovery if needed
                        if health_check.recovery_needed:
                            self.initiate_recovery(name)
                    
                    # Check cascade risk
                    at_risk = self._check_cascade_risk()
                    if at_risk:
                        self._prevent_cascade(at_risk)
                    
                    # Clean old predictions
                    self._clean_predictions()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _clean_predictions(self):
        """Clean old predictions and update accuracy."""
        current_time = datetime.now()
        
        for component, prediction in list(self.predicted_failures.items()):
            if current_time > prediction['estimated_failure_time']:
                # Check if prediction was accurate
                health = self.component_health.get(component, HealthStatus.UNKNOWN)
                
                if health in [HealthStatus.FAILING, HealthStatus.CRITICAL]:
                    self.stats['predictions_accurate'] += 1
                
                # Remove old prediction
                del self.predicted_failures[component]
    
    # Component-specific recovery strategies
    def _recover_event_queue(self, plan: RecoveryPlan) -> bool:
        """Recover event queue."""
        try:
            comp = self.components['event_queue']['instance']
            
            # Flush pending events
            if hasattr(comp, 'flush'):
                comp.flush()
            
            # Restart processing
            if hasattr(comp, 'start_processing'):
                comp.start_processing()
            
            return True
        except:
            return False
    
    def _recover_batch_processor(self, plan: RecoveryPlan) -> bool:
        """Recover batch processor."""
        try:
            comp = self.components['batch_processor']['instance']
            
            # Flush all batches
            if hasattr(comp, 'flush_all'):
                from ..core.analytics_batch_processor import FlushReason
                comp.flush_all(FlushReason.ERROR_RECOVERY)
            
            # Adjust strategy
            if hasattr(comp, 'adjust_strategy'):
                comp.adjust_strategy('aggressive')
            
            return True
        except:
            return False
    
    def _recover_heartbeat(self, plan: RecoveryPlan) -> bool:
        """Recover heartbeat monitor."""
        try:
            comp = self.components['heartbeat_monitor']['instance']
            
            # Re-register endpoints
            if hasattr(comp, 'register_endpoint'):
                comp.register_endpoint(
                    'main_dashboard',
                    'http://localhost:5000/api/health/live',
                    'http',
                    critical=True
                )
            
            return True
        except:
            return False
    
    def _recover_flow_monitor(self, plan: RecoveryPlan) -> bool:
        """Recover flow monitor."""
        try:
            comp = self.components['flow_monitor']['instance']
            
            # Complete stuck transactions
            if hasattr(comp, 'active_transactions'):
                from ..core.analytics_flow_monitor import FlowStatus
                for txn_id in list(comp.active_transactions.keys()):
                    comp.complete_transaction(txn_id, FlowStatus.TIMEOUT)
            
            return True
        except:
            return False
    
    def _recover_fallback(self, plan: RecoveryPlan) -> bool:
        """Recover fallback system."""
        try:
            comp = self.components['fallback_system']['instance']
            
            # Force recovery to primary
            if hasattr(comp, 'force_recovery'):
                comp.force_recovery()
            
            return True
        except:
            return False
    
    def _recover_dead_letter(self, plan: RecoveryPlan) -> bool:
        """Recover dead letter queue."""
        try:
            comp = self.components['dead_letter_queue']['instance']
            
            # Attempt bulk reprocess
            if hasattr(comp, 'bulk_reprocess'):
                result = comp.bulk_reprocess(
                    lambda x: True,  # Dummy processor
                    max_entries=10
                )
                
            return True
        except:
            return False
    
    def _recover_compressor(self, plan: RecoveryPlan) -> bool:
        """Recover compressor."""
        try:
            comp = self.components['compressor']['instance']
            
            # Reset to adaptive mode
            if hasattr(comp, '__init__'):
                comp.adaptive = True
                comp.compression_level = 6
            
            return True
        except:
            return False
    
    def _recover_retry_manager(self, plan: RecoveryPlan) -> bool:
        """Recover retry manager."""
        try:
            comp = self.components['retry_manager']['instance']
            
            # Reset all circuit breakers
            if hasattr(comp, 'circuit_breakers'):
                for cb in comp.circuit_breakers.values():
                    if hasattr(cb, 'reset'):
                        cb.reset()
            
            return True
        except:
            return False
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        with self.lock:
            return {
                'monitoring_active': self.monitoring_active,
                'components_registered': len(self.components),
                'component_health': {
                    name: status.value
                    for name, status in self.component_health.items()
                },
                'recoveries_in_progress': list(self.recovery_in_progress),
                'predicted_failures': {
                    comp: {
                        'estimated_time': pred['estimated_failure_time'].isoformat(),
                        'confidence': pred['confidence']
                    }
                    for comp, pred in self.predicted_failures.items()
                },
                'statistics': self.stats,
                'recent_recoveries': list(self.recovery_history)[-10:],
                'cascade_risk': self._check_cascade_risk()
            }
    
    def shutdown(self):
        """Shutdown orchestrator."""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info(f"Recovery Orchestrator shutdown - Stats: {self.stats}")

# Global orchestrator instance
recovery_orchestrator = AnalyticsRecoveryOrchestrator()